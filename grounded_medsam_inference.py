#%%
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import os, json
import pandas as pd
import tqdm
# %%
def dice_binary(preds, targets, smooth=1.0):
    assert preds.shape == targets.shape
    pred = preds == 1  # Assuming the foreground class is labeled as 1
    target = targets == 1
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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

# %%
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")
DEVICE0 = torch.device('cuda:0')
DEVICE1 = torch.device('cuda:1')
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/mnt/hanoverdev/models/BiomedSEEM/groundingdino/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_b"
# SAM_CHECKPOINT_PATH = "/mnt/hanoverdev/models/BiomedSEEM/medsam/medsam_vit_b.pth"
SAM_CHECKPOINT_PATH = "/mnt/hanoverdev/models/BiomedSEEM/sam/sam_vit_b_01ec64.pth"
sam_model_name = "medsam" if 'medsam' in SAM_CHECKPOINT_PATH else os.path.basename(SAM_CHECKPOINT_PATH)

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE0)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE1)
sam_predictor = SamPredictor(sam)
# %%
# inference on biomedseg data
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

def setup(task_dir, is_illegal=False):
    test_json = os.path.join(task_dir, "test_illegal.json") if is_illegal else os.path.join(task_dir, "test.json")
    test_image_dir = os.path.join(task_dir, "test")
    test_mask_dir = os.path.join(task_dir, "test_mask")
    output_dir = os.path.join(task_dir, f"test_groundingdino_{sam_model_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    groundedsam_output_dir = os.path.join(output_dir, f"anno_groundingdino_{sam_model_name}")
    grounding_dino_output_dir = os.path.join(output_dir, "anno_groundingdino")
    mask_output_dir = os.path.join(output_dir, f"mask_groundingdino_{sam_model_name}")
    if not os.path.exists(grounding_dino_output_dir):
        os.makedirs(grounding_dino_output_dir) 
    if not os.path.exists(groundedsam_output_dir):
        os.makedirs(groundedsam_output_dir)
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
    return test_json, test_image_dir, test_mask_dir, output_dir, groundedsam_output_dir, grounding_dino_output_dir, mask_output_dir

def inference(task_dir, sam_predictor, grounding_dino_model, run_grounding_dino_sam=True, save_mask=True, is_illegal=False):
    test_json, test_image_dir, test_mask_dir, output_dir, groundedsam_output_dir, grounding_dino_output_dir, mask_output_dir = setup(task_dir, is_illegal=is_illegal)
    # load test json
    try:
        with open(test_json, "r") as f:
            test_data = json.load(f)
    except:
        # todo: hack, json loading issue
        print(f"Error loading {test_json}")
        with open(test_json.replace("test.json", "test copy.json"), "r") as f:
            test_data = json.load(f)

    annotations = test_data["annotations"]
    # iterate over the test data
    bbox_grounding_dino_json = []
    dice_scores = []
    for line in tqdm.tqdm(annotations):
        image_path = os.path.join(test_image_dir, line["file_name"])
        mask_path = os.path.join(test_mask_dir, line["mask_file"])
        classes = [line["sentences"][0]['sent']] #todo: need to check if this is the right way to get the class
        gold_box = line["bbox"]
        # load image
        image = cv2.imread(image_path)
        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ , _
            in detections]
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    
        # # save grounding dino annotated image to output dir subfolder "groundingdino"
        # grounding_dino_output_image_path = os.path.join(grounding_dino_output_dir, os.path.basename(mask_path))
        # cv2.imwrite(grounding_dino_output_image_path, annotated_frame)
    
        # keep the best box
        if detections.confidence.size == 0:
            print(f"No detections for {mask_path}")
            # make placeholder or zero mask for the image
            best_box_idx = [0]
            detections.xyxy = np.array([[0, 0, 0, 0]])
            detections.confidence = np.array([0])
            detections.class_id = np.array([0])
        else:
            best_box_idx = [np.argmax(detections.confidence)]
            detections.xyxy = detections.xyxy[best_box_idx]
            detections.confidence = detections.confidence[best_box_idx]
            detections.class_id = detections.class_id[best_box_idx]
        labels = [f"{classes[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # cv2.imwrite(grounding_dino_output_image_path.replace(".png", "_best.png"), annotated_frame)
        # save the best box
        bbox_grounding_dino_json.append({
            "file_name": os.path.basename(image_path),
            "mask_file": os.path.basename(mask_path),
            "bbox_gold": gold_box,
            "bbox_pred": np.array(list(detections.xyxy[0])).tolist(),
            "sentences": line["sentences"],
            "class_id": int(detections.class_id[0]),
            "confidence": float(detections.confidence[0]) # convert to float
        })
       
        if run_grounding_dino_sam:
        # convert detections to masks
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            if save_mask:
            # annotate image with detections
                mask_annotator = sv.MaskAnnotator()

                mask_output_path = os.path.join(mask_output_dir, os.path.basename(mask_path))
                # save mask of black and white
                mask_image = np.zeros_like(image)
                mask_image[detections.mask[0]] = 255
                cv2.imwrite(mask_output_path, mask_image)
            #
            # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            # groundedsam_output_image_path = os.path.join(groundedsam_output_dir, os.path.basename(mask_path))
            # cv2.imwrite(groundedsam_output_image_path, annotated_image)
            
            # convert mask to binary and calculate dice
            gold_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gold_mask = gold_mask > 0
            dice = dice_binary(detections.mask[0], gold_mask)
            dice_scores.append({'image': os.path.basename(mask_path), 'dice': dice})

    # save dice_scores to csv
    if run_grounding_dino_sam:
        df = pd.DataFrame(dice_scores)
        df.to_csv(os.path.join(output_dir, "test_groundingdino_medsam_dice.csv"), index=False)
  
    # save box as csv
    df = pd.DataFrame(bbox_grounding_dino_json)
    output_file_name = "test_groundingdino_medsam_bbox_illegal.csv" if is_illegal else "test_groundingdino_medsam_bbox.csv"
    df.to_csv(os.path.join(output_dir, output_file_name), index=False)
# %%
if __name__ == "__main__":
    base_dir = "/mnt/hanoverdev/data/BiomedSeg"
    tasks = ['ACDC', 'AbdomenCT-1K', 'BreastCancerCellSegmentation', 'BreastUS', 'CAMUS', 'CDD-CESM', 'COVID-QU-Ex', 'CXR_Masks_and_Labels', 'FH-PS-AOP', 'G1020', 'GlaS', 'ISIC', 'LGG', 'LIDC-IDRI', 'MMs', 'NeoPolyp', 'OCT-CME', 'PROMISE12', 'PolypGen', 'QaTa-COV19', 'REFUGE', 'UWaterlooSkinCancer', 'kits23', 'siim-acr-pneumothorax', 'MSD/Task03_Liver', 'MSD/Task06_Lung', 'MSD/Task09_Spleen', 'MSD/Task04_Hippocampus', 'MSD/Task07_Pancreas', 'MSD/Task10_Colon', 'MSD/Task02_Heart', 'MSD/Task05_Prostate', 'MSD/Task08_HepaticVessel', 'Radiography/COVID', 'Radiography/Lung_Opacity', 'Radiography/Normal', 'Radiography/Viral_Pneumonia', 'amos22/CT', 'amos22/MRI', 'PanNuke', 'BTCV-Cervix', 'COVID-19_CT', 'MSD/Task01_BrainTumour']
    # tasks = ['PanNuke']
    print("Loading...")
    for task in tasks:
        task_dir = os.path.join(base_dir, task)
        print(f"Processing {task}")
        inference(task_dir, sam_predictor, grounding_dino_model, save_mask=True)
        
    # illega text experiment
    # base_dir = "/mnt/hanoverdev/data/BiomedSeg"
    # tasks_illegal = []
    # task_legal = ['ACDC', 'Radiography/Normal', 'kits23', 'BreastUS', 'GlaS', 'ISIC','REFUGE']
    # print('processing illegal cases')
    # for task in tasks_illegal:
    #     task_dir = os.path.join(base_dir, task)
    #     print(f"Processing {task}")
    #     inference(task_dir, sam_predictor, grounding_dino_model, run_grounding_dino_sam=False, save_mask=False, is_illegal=True)
    # print('processing legal cases')
    # for task in task_legal:
    #     task_dir = os.path.join(base_dir, task)
    #     print(f"Processing {task}")
    #     inference(task_dir, sam_predictor, grounding_dino_model, run_grounding_dino_sam=False, save_mask=False, is_illegal=False)