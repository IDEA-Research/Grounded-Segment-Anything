import data
import cv2
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from models import imagebind_model
from models.imagebind_model import ModalityType

from segment_anything import build_sam, SamAutomaticMaskGenerator

from utils import (
    segment_image, 
    convert_box_xywh_to_xyxy,
    get_indices_of_values_above_threshold,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


"""
Step 1: Instantiate model
"""
# Segment Anything
mask_generator = SamAutomaticMaskGenerator(
    build_sam(checkpoint=".checkpoints/sam_vit_h_4b8939.pth").to(device),
    points_per_side=16,
)

# ImageBind
bind_model = imagebind_model.imagebind_huge(pretrained=True)
bind_model.eval()
bind_model.to(device)


"""
Step 2: Generate auto masks with SAM
"""
image_path = ".assets/car_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)


"""
Step 3: Get cropped images based on mask and box
"""
cropped_boxes = []
image = Image.open(image_path)
for mask in tqdm(masks):
    cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))


"""
Step 4: Run ImageBind model to get similarity between cropped image and different modalities
"""
# load referring image
referring_image_path = ".assets/referring_car_image.jpg"
referring_image = Image.open(referring_image_path)

image_list = []
image_list += cropped_boxes
image_list.append(referring_image)

def retriev_vision_and_vision(elements):
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data_from_pil_image(elements, device),
    }
    with torch.no_grad():
        embeddings = bind_model(inputs)

    # cropped box region embeddings
    cropped_box_embeddings = embeddings[ModalityType.VISION][:-1, :]
    referring_image_embeddings = embeddings[ModalityType.VISION][-1, :]

    vision_referring_result = torch.softmax(cropped_box_embeddings @ referring_image_embeddings.T, dim=0),
    return vision_referring_result  # [113, 1]


vision_referring_result = retriev_vision_and_vision(image_list)


"""
Step 5: Merge the top similarity masks to get the final mask and save the merged mask

Image / Text mask
"""

# get highest similar mask with threshold
# result[0] shape: [113, 1]
threshold = 0.017
index = get_indices_of_values_above_threshold(vision_referring_result[0], threshold)


segmentation_masks = []
for seg_idx in index:
    segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
    segmentation_masks.append(segmentation_mask_image)

original_image = Image.open(image_path)
overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
overlay_color = (255, 255, 255, 0)

draw = ImageDraw.Draw(overlay_image)
for segmentation_mask_image in segmentation_masks:
    draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

# return Image.alpha_composite(original_image.convert('RGBA'), overlay_image) 
mask_image = overlay_image.convert("RGB")
mask_image.save("./image_referring_sam_merged_mask.jpg")
