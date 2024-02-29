import cv2
import numpy as np
import supervision as sv
from typing import List
from PIL import Image

import torch

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# Tag2Text
# from ram.models import tag2text_caption
from ram.models import ram
# from ram import inference_tag2text
from ram import inference_ram
import torchvision
import torchvision.transforms as TS


# Hyper-Params
SOURCE_IMAGE_PATH = "./assets/demo9.jpg"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

TAG2TEXT_CHECKPOINT_PATH = "./tag2text_swin_14m.pth"
RAM_CHECKPOINT_PATH = "./ram_swin_large_14m.pth"

TAG2TEXT_THRESHOLD = 0.64
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam_predictor = SamPredictor(sam)

# Tag2Text
# initialize Tag2Text
normalize = TS.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = TS.Compose(
    [
        TS.Resize((384, 384)),
        TS.ToTensor(), 
        normalize
    ]
)

DELETE_TAG_INDEX = []  # filter out attributes and action which are difficult to be grounded
for idx in range(3012, 3429):
    DELETE_TAG_INDEX.append(idx)

# tag2text_model = tag2text_caption(
#     pretrained=TAG2TEXT_CHECKPOINT_PATH,
#     image_size=384,
#     vit='swin_b',
#     delete_tag_index=DELETE_TAG_INDEX
# )
# # threshold for tagging
# # we reduce the threshold to obtain more tags
# tag2text_model.threshold = TAG2TEXT_THRESHOLD
# tag2text_model.eval()
# tag2text_model = tag2text_model.to(DEVICE)

ram_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                        image_size=384,
                                        vit='swin_l')
ram_model.eval()
ram_model = ram_model.to(DEVICE)

# load image
image = cv2.imread(SOURCE_IMAGE_PATH)  # bgr
image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # rgb

image_pillow = image_pillow.resize((384, 384))
image_pillow = transform(image_pillow).unsqueeze(0).to(DEVICE)

specified_tags='None'
# res = inference_tag2text(image_pillow , tag2text_model, specified_tags)
res = inference_ram(image_pillow , ram_model)

# Currently ", " is better for detecting single tags
# while ". " is a little worse in some case
AUTOMATIC_CLASSES=res[0].split(" | ")

print(f"Tags: {res[0].replace(' |', ',')}")


# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=AUTOMATIC_CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=BOX_THRESHOLD
)

# NMS post process
print(f"Before NMS: {len(detections.xyxy)} boxes")
nms_idx = torchvision.ops.nms(
    torch.from_numpy(detections.xyxy), 
    torch.from_numpy(detections.confidence), 
    IOU_THRESHOLD
).numpy().tolist()

detections.xyxy = detections.xyxy[nms_idx]
detections.confidence = detections.confidence[nms_idx]
detections.class_id = detections.class_id[nms_idx]

print(f"After NMS: {len(detections.xyxy)} boxes")

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{AUTOMATIC_CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# save the annotated grounding dino image
cv2.imwrite("groundingdino_auto_annotated_image.jpg", annotated_frame)

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{AUTOMATIC_CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save the annotated grounded-sam image
cv2.imwrite("ram_grounded_sam_auto_annotated_image.jpg", annotated_image)
