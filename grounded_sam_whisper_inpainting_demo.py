import argparse
import os
from warnings import warn

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

# whisper
import whisper

# ChatGPT
import openai


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

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
    print(load_res)
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
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)
    

def speech_recognition(speech_file, model):
    # whisper
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(speech_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    speech_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    speech_text = result.text
    return speech_text, speech_language


def filter_prompts_with_chatgpt(caption, max_tokens=100, model="gpt-3.5-turbo"):
    prompt = [
        {
            'role': 'system',
            'content': f"Extract the main object to be replaced and marked it as 'main_object', " + \
                       f"Extract the remaining part as 'other prompt' " + \
                       f"Return (main_object, other prompt)" + \
                       f'Given caption: {caption}.'
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    try:
        det_prompt, inpaint_prompt = reply.split('\n')[0].split(':')[-1].strip(), reply.split('\n')[1].split(':')[-1].strip()
    except:
        warn(f"Failed to extract tags from caption") # use caption as det_prompt, inpaint_prompt
        det_prompt, inpaint_prompt = caption, caption
    return det_prompt, inpaint_prompt


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--det_speech_file", type=str, help="grounding speech file")
    parser.add_argument("--inpaint_speech_file", type=str, help="inpaint speech file")
    parser.add_argument("--prompt_speech_file", type=str, help="prompt speech file, no need to provide det_speech_file")
    parser.add_argument("--enable_chatgpt", action="store_true", help="enable chatgpt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument("--whisper_model", type=str, default="small", help="whisper model version: tiny, base, small, medium, large")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--prompt_extra", type=str, default=" high resolution, real scene", help="extra prompt for inpaint")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image

    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    inpaint_mode = args.inpaint_mode
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
    
    # recognize speech
    whisper_model = whisper.load_model(args.whisper_model)
    
    if args.enable_chatgpt:
        openai.api_key = args.openai_key
        if args.openai_proxy:
            openai.proxy = {"http": args.openai_proxy, "https": args.openai_proxy}
        speech_text, _ = speech_recognition(args.prompt_speech_file, whisper_model)
        det_prompt, inpaint_prompt = filter_prompts_with_chatgpt(speech_text)
        inpaint_prompt += args.prompt_extra
        print(f"det_prompt: {det_prompt}, inpaint_prompt: {inpaint_prompt}")
    else:
        det_prompt, det_speech_language = speech_recognition(args.det_speech_file, whisper_model)
        inpaint_prompt, inpaint_speech_language = speech_recognition(args.inpaint_speech_file, whisper_model)
        print(f"det_prompt: {det_prompt}, using language: {det_speech_language}")
        print(f"inpaint_prompt: {inpaint_prompt}, using language: {inpaint_speech_language}")
    
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )

    # masks: [1, 1, 512, 512]

    # inpainting pipeline
    if inpaint_mode == 'merge':
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # prompt = "A sofa, high quality, detailed"
    image = pipe(prompt=inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
    image.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))

    # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

