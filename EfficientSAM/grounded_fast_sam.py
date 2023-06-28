import argparse
import cv2
from ultralytics import YOLO
from FastSAM.tools import *
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from torchvision.ops import box_convert
import ast

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./FastSAM/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument(
        "--text", type=str, default="the black dog.", help="text prompt for GroundingDINO"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[0,0,0,0]", help="[x,y,w,h]")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):

    # Image Path
    img_path = args.img_path
    text = args.text

    # path to save img
    save_path = args.output
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    basename = os.path.basename(args.img_path).split(".")[0]

    # Build Fast-SAM Model
    # ckpt_path = "/comp_robot/rentianhe/code/Grounded-Segment-Anything/FastSAM/FastSAM-x.pt"
    model = YOLO(args.model_path)

    results = model(
        args.img_path,
        imgsz=args.imgsz,
        device=args.device,
        retina_masks=args.retina,
        iou=args.iou,
        conf=args.conf,
        max_det=100,
    )


    # Build GroundingDINO Model
    groundingdino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_ckpt_path = "./groundingdino_swint_ogc.pth"

    image_source, image = load_image(img_path)
    model = load_model(groundingdino_config, groundingdino_ckpt_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=0.3,
        text_threshold=0.25,
        device=args.device,
    )


    # Grounded-Fast-SAM

    ori_img = cv2.imread(img_path)
    ori_h = ori_img.shape[0]
    ori_w = ori_img.shape[1]

    # Save each frame due to the post process from FastSAM
    boxes = boxes * torch.Tensor([ori_w, ori_h, ori_w, ori_h])
    print(f"Detected Boxes: {len(boxes)}")
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()
    for box_idx in range(len(boxes)):
        mask, _ = box_prompt(
            results[0].masks.data,
            boxes[box_idx],
            ori_h,
            ori_w,
        )
        annotations = np.array([mask])
        img_array = fast_process(
            annotations=annotations,
            args=args,
            mask_random_color=True,
            bbox=boxes[box_idx],
        )
        cv2.imwrite(os.path.join(save_path, basename + f"_{str(box_idx)}_caption_{phrases[box_idx]}.jpg"), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    args = parse_args()
    main(args)
