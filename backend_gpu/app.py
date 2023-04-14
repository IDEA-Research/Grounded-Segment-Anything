from flask import Flask, request, Response
import base64
import json
import tempfile
from grounded_sam_inpainting_demo import main as grounded_sam_inpainting
from pathlib import Path

app = Flask(__name__, static_url_path="")


def generate_img(hint: str, prompt: str, img: bytes) -> bytes:
    config_file = Path("GroundingDINO_SwinT_OGC.py")
    grounded_checkpoint = Path("groundingdino_swint_ogc.pth")
    sam_checkpoint = Path("sam_vit_h_4b8939.pth")
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.png"
        with open(input_path, "wb") as f:
            f.write(img)

        img_fname = grounded_sam_inpainting(
            config_file=str(config_file),
            grounded_checkpoint=grounded_checkpoint,
            sam_checkpoint=sam_checkpoint,
            image_path=input_path,
            det_prompt=hint,
            inpaint_prompt=prompt,
            output_dir=temp_dir,
            box_threshold=0.3,
            text_threshold=0.25,
            inpaint_mode="first",
            device="cuda",
        )

        with open(img_fname, "rb") as f:
            return f.read()


@app.route("/api/process", methods=["POST"])
def process():
    hint: str = request.json["hint"]
    prompt: str = request.json["prompt"]
    image_base64: str = request.json["imagebase64"]

    png_bytes: bytes = base64.b64decode(image_base64)
    png_result = generate_img(hint, prompt, png_bytes)
    png_result_base64 = base64.b64encode(png_result)

    return Response(
        response=json.dumps({"resultbase64": png_result_base64}),
        status=200,
    )

@app.route("/", methods=["GET"])
def hello():
    return "hello"