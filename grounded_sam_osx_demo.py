import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import argparse
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor


# OSX
import sys
sys.path.insert(0, 'grounded-sam-osx')
from osx import get_model
from config import cfg
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x

os.environ["PYOPENGL_PLATFORM"] = "egl"
from utils.vis import render_mesh, save_obj
cudnn.benchmark = True

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
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    if 'person' in label.lower() or 'human' in label.lower():
        color = 'green'
    else:
        color = 'blue'
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0-5, label, fontsize=5, color='white',bbox={'facecolor': color, 'alpha': 0.7, 'pad': 1, 'edgecolor': 'none'})

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

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
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def bbox_resize(bbox, scale=1.0):
    center = (bbox[2:] + bbox[:2]) / 2
    new_size = (bbox[2:] - bbox[:2]) * scale
    new_bbox = torch.cat((center - new_size / 2, center + new_size / 2))
    return new_bbox

def mesh_recovery(original_img, bboxes):
    transform = transforms.ToTensor()
    original_img_height, original_img_width = original_img.shape[:2]

    vis_img = original_img.copy()
    for bbox in bboxes:  # [x1, y1, x2, y2]
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]   # xyxy -> xyhw
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]

        # forward
        inputs = {'img': img}
        with torch.no_grad():
            out = model(inputs, 'test')
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

        # # save mesh
        # save_obj(mesh, smpl_x.face, 'output.obj')

        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
                   cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        rendered_img, _ = render_mesh(vis_img[:, :, ::-1], mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        vis_img = rendered_img.copy()

    return rendered_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--osx_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    osx_checkpoint = args.osx_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # initialize OSX
    model = get_model()
    model = DataParallel(model).cuda()
    ckpt = torch.load(osx_checkpoint)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # scale up the human bboxes
    boxes_human = []
    for i, label in enumerate(pred_phrases):
        if 'person' in label.lower() or 'human' in label.lower():
            boxes_filt[i] = bbox_resize(boxes_filt[i], scale=1.1)
            boxes_human.append(boxes_filt[i])

    # predict and visualize 3d human mesh
    for i, label in enumerate(pred_phrases):
        if 'person' in label.lower() or 'man' in label.lower():
            boxes_human.append(boxes_filt[i])
    rendered_img = mesh_recovery(image, boxes_human)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam_osx_output.jpg"), rendered_img)

    # draw output image
    fig, (plt1, plt2) = plt.subplots(ncols=2, figsize=(10, 20), gridspec_kw={'wspace':0, 'hspace':0})

    plt1.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt1, random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt1, label)
    rendered_img = cv2.imread(os.path.join(output_dir, "grounded_sam_osx_output.jpg"))
    plt2.imshow(rendered_img)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt2, label)
    plt1.axis('off')
    plt2.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_osx_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)


