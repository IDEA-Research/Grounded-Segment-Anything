import os
import random

import gradio as gr
import argparse

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np

# diffusers
import torch
from diffusers import StableDiffusionInpaintPipeline

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration


def generate_caption(processor, blip_model, raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def transform_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=2)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)



config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint='sam_vit_h_4b8939.pth' 
output_dir="outputs"
device="cuda"


blip_processor = None
blip_model = None
groundingdino_model = None
sam_predictor = None
inpaint_pipeline = None

def run_grounded_sam(input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode):

    global blip_processor, blip_model, groundingdino_model, sam_predictor, inpaint_pipeline

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil = input_image.convert("RGB")
    transformed_image = transform_image(image_pil)

    if groundingdino_model is None:
        groundingdino_model = load_model(config_file, ckpt_filenmae, device=device)

    if task_type == 'automatic':
        # generate caption and tags
        # use Tag2Text can generate better captions
        # https://huggingface.co/spaces/xinyu1205/Tag2Text
        # but there are some bugs...
        blip_processor = blip_processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = blip_model or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        text_prompt = generate_caption(blip_processor, blip_model, image_pil)
        print(f"Caption: {text_prompt}")

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
    )

    size = image_pil.size

    # process boxes
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()


    if task_type == 'seg' or task_type == 'inpainting' or task_type == 'automatic':
        if sam_predictor is None:
            # initialize SAM
            assert sam_checkpoint, 'sam_checkpoint is not found!'
            sam = build_sam(checkpoint=sam_checkpoint)
            sam.to(device=device)
            sam_predictor = SamPredictor(sam)

        image = np.array(image_pil)
        sam_predictor.set_image(image)

        if task_type == 'automatic':
            # use NMS to handle overlapped boxes
            print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            print(f"After NMS: {boxes_filt.shape[0]} boxes")
            print(f"Revise caption with number: {text_prompt}")

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )

        # masks: [1, 1, 512, 512]

    if task_type == 'det':
        image_draw = ImageDraw.Draw(image_pil)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)

        return [image_pil]
    elif task_type == 'seg' or task_type == 'automatic':
        
        mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))

        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

        image_draw = ImageDraw.Draw(image_pil)

        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)

        if task_type == 'automatic':
            image_draw.text((10, 10), text_prompt, fill='black')

        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]
    elif task_type == 'inpainting':
        assert inpaint_prompt, 'inpaint_prompt is not found!'
        # inpainting pipeline
        if inpaint_mode == 'merge':
            masks = torch.sum(masks, dim=0).unsqueeze(0)
            masks = torch.where(masks > 0, True, False)
        mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
        mask_pil = Image.fromarray(mask)
        
        if inpaint_pipeline is None:
            inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
            )
            inpaint_pipeline = inpaint_pipeline.to("cuda")

        image = inpaint_pipeline(prompt=inpaint_prompt, image=image_pil.resize((512, 512)), mask_image=mask_pil.resize((512, 512))).images[0]
        image = image.resize(size)

        return [image, mask_pil]
    else:
        print("task_type:{} error!".format(task_type))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument('--port', type=int, default=7589, help='port to run the server')
    parser.add_argument('--no-gradio-queue', action="store_true", help='path to the SAM checkpoint')
    args = parser.parse_args()

    print(args)

    block = gr.Blocks()
    if not args.no_gradio_queue:
        block = block.queue()

    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil", value="assets/demo1.jpg")
                task_type = gr.Dropdown(["det", "seg", "inpainting", "automatic"], value="automatic", label="task_type")
                text_prompt = gr.Textbox(label="Text Prompt")
                inpaint_prompt = gr.Textbox(label="Inpaint Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.001
                    )
                    inpaint_mode = gr.Dropdown(["merge", "first"], value="merge", label="inpaint_mode")

            with gr.Column():
                gallery = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery"
                ).style(preview=True, grid=2, object_fit="scale-down")

        run_button.click(fn=run_grounded_sam, inputs=[
                        input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode], outputs=gallery)


    block.launch(server_name='0.0.0.0', server_port=args.port, debug=args.debug, share=args.share)