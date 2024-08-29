import argparse
import os
import sys
import time
import torch
import numpy as np
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything imports
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    print("Loading model from...........", device)
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    
    # Load the model checkpoint onto the specific GPU
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # Keep it on the device
    boxes = outputs["pred_boxes"][0]  # Keep it on the device

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases


def process_image(image_path, model, predictor, output_dir, text_prompt, box_threshold, text_threshold, device):
    
    # Load the image and move to GPU
    image_pil, image = load_image(image_path)
    # image_pil.save(os.path.join(output_dir, f"raw_image_{os.path.basename(image_path)}.jpg"))
    # Run GroundingDINO model to get bounding boxes and labels
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # Load SAM model onto GPU    
    image_cv = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)

    # Convert boxes to original image size
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.tensor([W, H, W, H], device=device)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    # Transform boxes to be compatible with SAM
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)

    # Get masks using SAM
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # Visualization and saving
    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.cpu().numpy(), plt.gca(), label)
    image_base_name = os.path.basename(image_path).split('.')[0]
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_{image_base_name}.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    plt.close()

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases, image_base_name)
    # Clear GPU memory
    del image, transformed_boxes, masks  # model, sam
    # torch.cuda.empty_cache()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # print("mask.shape:", mask.shape)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, image_base_name=''):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:], device=mask_list.device)
    for idx, mask in enumerate(mask_list):
        mask_img[mask[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{image_base_name}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.cpu().numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'{image_base_name}.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--input_path", type=str, required=True, help="path to directory containing image files")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the inference on, e.g., 'cuda' or 'cuda:0'")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True

    start_time = time.time()
    # Determine if we are using a single GPU or all available GPUs
    if args.device == "cuda":
        if torch.cuda.device_count() > 1:
            device_list = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]  # Use all GPUs
        else:
            device_list = [torch.device("cuda:0")]  # Default to first GPU
    else:
        device_list = [torch.device(args.device)]
    print("device_list:", device_list)

    # Get list of images
    image_paths = [os.path.join(args.input_path, img) for img in os.listdir(args.input_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Split images among available GPUs
    image_batches = np.array_split(image_paths, len(device_list))
    print("Processing images:", image_batches)
    # Function to process a batch of images on the specified device
    def process_batch(batch_images, model_config, model_checkpoint, sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, device, output_dir):
        # Load model onto GPU
        torch.cuda.set_device(device)
        model = load_model(model_config, model_checkpoint, device)
        
        # Load SAM model onto GPU
        if use_sam_hq:
            sam = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)
        else:
            sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
        # Move model to the correct device
        device = torch.device(device)
        model.to(device)
        sam.to(device)
        predictor = SamPredictor(sam)
        for image_path in batch_images:
            # Process each image
            print("Processing image:", image_path)
            process_image(
                image_path=image_path,
                model=model,
                predictor=predictor,
                output_dir=output_dir,
                text_prompt=args.text_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=device
            )
            print("Image processing complete {}".format(image_path))
        # Clear GPU memory after processing the batch
        # del model, sam
        torch.cuda.empty_cache()

    # Use ThreadPoolExecutor to parallelize the processing across GPUs
    with ThreadPoolExecutor(max_workers=len(device_list)*2) as executor:
        futures = []
        for i, device in enumerate(device_list):
            print(f"Processing images on device {device}")
            print("Image batches for each GPU:", len(image_batches[i]))
            futures.append(executor.submit(
                process_batch, image_batches[i], args.config, args.grounded_checkpoint, args.sam_version, args.sam_checkpoint, args.sam_hq_checkpoint, args.use_sam_hq, device, args.output_dir
            ))

        # Wait for all threads to complete
        for future in futures:
            future.result()

    print("Processing complete. Results saved to the output directory.")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")