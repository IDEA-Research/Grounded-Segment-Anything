import gradio as gr
import json
import argparse
import os
import copy

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import openai
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from transformers import BlipProcessor, BlipForConditionalGeneration
# segment anything
from segment_anything import build_sam, SamPredictor 
from segment_anything.utils.amg import remove_small_regions
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from huggingface_hub import hf_hub_download
from sys import platform

#macos
if platform == 'darwin':
    import matplotlib
    matplotlib.use('agg')

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_image(image_path):
    # # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image
    image_pil = image_path

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    mask_img_path = os.path.join(output_dir, 'mask.jpg')
    plt.savefig(mask_img_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    
    mask_json_path = os.path.join(output_dir, 'mask.json')
    with open(mask_json_path, 'w') as f:
        json.dump(json_data, f)

    return mask_img_path, mask_json_path

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint='sam_vit_h_4b8939.pth'
output_dir="outputs"
device="cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_tags(caption, split=',', max_tokens=100, model="gpt-3.5-turbo", openai_key=''):
    openai.api_key = openai_key
    prompt = [
        {
            'role': 'system',
            'content': 'Extract the unique nouns in the caption. Remove all the adjectives. ' + \
                       f'List the nouns in singular form. Split them by "{split} ". ' + \
                       f'Caption: {caption}.'
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    # sometimes return with "noun: xxx, xxx, xxx"
    tags = reply.split(':')[-1].strip()
    return tags

def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    prompt = [
        {
            'role': 'system',
            'content': 'Revise the number in the caption if it is wrong. ' + \
                       f'Caption: {caption}. ' + \
                       f'True object number: {object_num}. ' + \
                       'Only give the revised caption: '
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    # sometimes return with "Caption: xxx, xxx, xxx"
    caption = reply.split(':')[-1].strip()
    return caption

def run_grounded_sam(image_path, openai_key, box_threshold, text_threshold, iou_threshold, area_threshold):
    assert openai_key, 'Openai key is not found!'

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path.convert("RGB"))
    # load model
    model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    caption = generate_caption(image_pil)
    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    split = ','
    tags = generate_tags(caption, split=split, openai_key=openai_key)

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )

    size = image_pil.size

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = np.array(image_path)
    predictor.set_image(image)

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    caption = check_caption(caption, pred_phrases)
    print(f"Revise caption with number: {caption}")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    # area threshold: remove the mask when area < area_thresh (in pixels)
    new_masks = []
    for mask in masks:
        # reshape to be used in remove_small_regions()
        mask = mask.cpu().numpy().squeeze()
        mask, _ = remove_small_regions(mask, area_threshold, mode="holes")
        mask, _ = remove_small_regions(mask, area_threshold, mode="islands")
        new_masks.append(torch.as_tensor(mask).unsqueeze(0))

    masks = torch.stack(new_masks, dim=0)
    # masks: [1, 1, 512, 512]
    assert sam_checkpoint, 'sam_checkpoint is not found!'

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    image_path = os.path.join(output_dir, "grounding_dino_output.jpg")
    plt.savefig(image_path, bbox_inches="tight")
    image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    mask_img_path, _ = save_mask_data('./outputs', masks, boxes_filt, pred_phrases)

    mask_img = cv2.cvtColor(cv2.imread(mask_img_path), cv2.COLOR_BGR2RGB)

    return image_result, mask_img, caption, tags

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil")
                openai_key = gr.Textbox(label="OpenAI key")

                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    iou_threshold = gr.Slider(
                        label="IoU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.001
                    )
                    area_threshold = gr.Slider(
                        label="Area Threshold", minimum=0.0, maximum=2500, value=100, step=10
                    )

            with gr.Column():
                image_caption = gr.Textbox(label="Image Caption")
                identified_labels = gr.Textbox(label="Key objects extracted by ChatGPT")
                gallery = gr.outputs.Image(
                    type="pil",
                ).style(full_width=True, full_height=True)

                mask_gallary = gr.outputs.Image(
                    type="pil",
                ).style(full_width=True, full_height=True)


        run_button.click(fn=run_grounded_sam, inputs=[
                        input_image, openai_key, box_threshold, text_threshold, iou_threshold, area_threshold], 
                        outputs=[gallery, mask_gallary, image_caption, identified_labels])


    block.launch(server_name='0.0.0.0', server_port=7589, debug=args.debug, share=args.share)