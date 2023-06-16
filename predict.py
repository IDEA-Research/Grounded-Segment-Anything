# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import json
from typing import Any
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel

from subprocess import call

HOME = os.getcwd()
os.chdir("GroundingDINO")
call("pip install -q .", shell=True)
os.chdir(HOME)
os.chdir("segment_anything")
call("pip install -q .", shell=True)
os.chdir(HOME)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# segment anything
from segment_anything import build_sam, build_sam_hq, SamPredictor

import sys

sys.path.append("Tag2Text")
from models.tag2text import ram


class ModelOutput(BaseModel):
    tags: str
    rounding_box_img: Path
    masked_img: Path
    json_data: Any


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.image_size = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # load model
        self.ram_model = ram(
            pretrained="pretrained/ram_swin_large_14m.pth",
            image_size=self.image_size,
            vit="swin_l",
        )
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.device)

        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "pretrained/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        self.sam = SamPredictor(
            build_sam(checkpoint="pretrained/sam_vit_h_4b8939.pth").to(self.device)
        )
        self.sam_hq = SamPredictor(
            build_sam_hq(checkpoint="pretrained/sam_hq_vit_h.pth").to(self.device)
        )

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        use_sam_hq: bool = Input(
            description="Use sam_hq instead of SAM for prediction", default=False
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        # default settings
        box_threshold = 0.25
        text_threshold = 0.2
        iou_threshold = 0.5

        image_pil, image = load_image(str(input_image))

        raw_image = image_pil.resize((self.image_size, self.image_size))
        raw_image = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tags, tags_chinese = self.ram_model.generate_tag(raw_image)

        tags = tags[0].replace(" |", ",")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.model, image, tags, box_threshold, text_threshold, device=self.device
        )

        predictor = self.sam_hq if use_sam_hq else self.sam

        image = cv2.imread(str(input_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = (
            torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        )
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        # draw output image
        plt.figure(figsize=(10, 10))
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        rounding_box_path = "/tmp/automatic_label_output.png"
        plt.axis("off")
        plt.savefig(
            Path(rounding_box_path), bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.close()

        # save masks and json data
        value = 0  # 0 for background
        mask_img = torch.zeros(masks.shape[-2:])
        for idx, mask in enumerate(masks):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis("off")
        masks_path = "/tmp/mask.png"
        plt.savefig(masks_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()

        json_data = {
            "tags": tags,
            "mask": [{"value": value, "label": "background"}],
        }
        for label, box in zip(pred_phrases, boxes_filt):
            value += 1
            name, logit = label.split("(")
            logit = logit[:-1]  # the last is ')'
            json_data["mask"].append(
                {
                    "value": value,
                    "label": name,
                    "logit": float(logit),
                    "box": box.numpy().tolist(),
                }
            )

        json_path = "/tmp/label.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        return ModelOutput(
            tags=tags,
            masked_img=Path(masks_path),
            rounding_box_img=Path(rounding_box_path),
            json_data=Path(json_path),
        )


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, device="cpu"
):
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
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


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
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=1.5)
    )
    ax.text(x0, y0, label)
