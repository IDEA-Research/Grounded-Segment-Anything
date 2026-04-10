import argparse
import gc
import inspect
import os
import random
import re
import sys
import threading
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torchvision
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError as DeepTranslatorRequestError
from deep_translator.exceptions import TooManyRequests, TranslationNotFound
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
try:
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, build_sam
except ImportError:
    from segment_anything.segment_anything import SamAutomaticMaskGenerator, SamPredictor, build_sam
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

try:
    from transformers import GroundingDinoProcessor
except ImportError:
    GroundingDinoProcessor = AutoProcessor

try:
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

    LOCAL_GROUNDING_IMPORT_ERROR = None
except Exception as exc:
    T = None
    build_model = None
    SLConfig = None
    clean_state_dict = None
    get_phrases_from_posmap = None
    LOCAL_GROUNDING_IMPORT_ERROR = exc


CONFIG_FILE = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
LOCAL_GROUNDING_CHECKPOINT = "groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
HF_GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
INPAINT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
OUTPUT_DIR = "outputs"

DEVICE = "cuda"
MAIN_DTYPE = torch.float16
BLIP_DTYPE = torch.float16
SD_DTYPE = torch.float16
MAX_INPAINT_SIZE = 512
DEFAULT_INPAINT_STEPS = 24
MAX_GRADIO_CONCURRENCY = 1
TRANSLATION_TIMEOUT_SECONDS = 6.0
ZH_TEXT_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
GRADIO_BLOCKS_ACCEPTS_CSS = "css" in inspect.signature(gr.Blocks).parameters
GRADIO_LAUNCH_ACCEPTS_CSS = "css" in inspect.signature(gr.Blocks.launch).parameters
GRADIO_IMAGE_ACCEPTS_SOURCE = "source" in inspect.signature(gr.Image).parameters
GRADIO_IMAGE_ACCEPTS_TOOL = "tool" in inspect.signature(gr.Image).parameters
GRADIO_QUEUE_ACCEPTS_CONCURRENCY_COUNT = "concurrency_count" in inspect.signature(gr.Blocks.queue).parameters

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

APP_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(65, 123, 255, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(20, 184, 166, 0.18), transparent 25%),
        linear-gradient(180deg, #08111f 0%, #0d1729 52%, #101b30 100%);
    color: #e5eefb;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.hero-card,
.panel-card,
.status-card {
    background: rgba(9, 20, 39, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 24px;
    box-shadow: 0 24px 70px rgba(2, 8, 23, 0.35);
}
.hero-card {
    padding: 28px 30px;
    margin-bottom: 18px;
}
.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.16);
    color: #8fe7ff;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero-title {
    margin: 14px 0 12px;
    font-size: 34px;
    line-height: 1.2;
    font-weight: 700;
    color: #f8fbff;
}
.hero-text {
    margin: 0;
    font-size: 15px;
    line-height: 1.7;
    color: #c7d6ea;
    max-width: 980px;
}
.panel-card {
    padding: 20px;
}
.panel-title {
    margin: 0 0 6px;
    font-size: 22px;
    font-weight: 700;
    color: #f8fbff;
}
.panel-text {
    margin: 0 0 18px;
    color: #9eb1c9;
    line-height: 1.6;
    font-size: 14px;
}
.status-card {
    padding: 16px 18px;
    margin-top: 18px;
}
#run-btn {
    min-height: 52px;
    font-size: 16px;
    font-weight: 700;
    border-radius: 16px;
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 50%, #14b8a6 100%);
    border: none;
}
#result-image,
#mask-image {
    border-radius: 18px;
    overflow: hidden;
}
#stop-btn {
    min-height: 52px;
    font-size: 16px;
    font-weight: 700;
    border-radius: 16px;
}
"""

MODEL_STATE: Dict[str, Any] = {
    "grounding_backend": None,
    "sam_model": None,
    "sam_predictor": None,
    "sam_automask_generator": None,
    "blip_processor": None,
    "blip_model": None,
    "inpaint_pipeline": None,
}
CANCEL_FLAGS: Dict[str, bool] = {}
CANCEL_FLAGS_LOCK = threading.Lock()

def ensure_cuda_available() -> None:
    if DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "当前策略要求 GPU 运行，但本环境未检测到可用 CUDA。"
            "请改用 Python 3.10 的 GPU 环境启动，例如 "
            "`\\.venv310-gpu\\Scripts\\python.exe gradio_app.py`，"
            "并先运行 `scripts/check_blackwell_cuda.py` 完成 CUDA 自检。"
        )


def autocast_context():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=MAIN_DTYPE)
    return nullcontext()


def cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def validate_runtime_environment() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "当前启动环境不是 Python 3.10。"
            f"检测到的是 Python {sys.version_info.major}.{sys.version_info.minor}，"
            "而这套 Grounded-SAM GPU 依赖与当前项目自检通过的环境是 Python 3.10。"
            "请改用 `\\.venv310-gpu\\Scripts\\python.exe gradio_app.py` 启动。"
        )
    ensure_cuda_available()


class CancelledByUser(Exception):
    pass


class PromptTranslationError(RuntimeError):
    pass


class PromptTranslationTimeoutError(PromptTranslationError):
    pass


def set_cancel_flag(session_id: str, value: bool) -> None:
    if not session_id:
        return
    with CANCEL_FLAGS_LOCK:
        CANCEL_FLAGS[session_id] = value


def clear_cancel_flag(session_id: str) -> None:
    if not session_id:
        return
    with CANCEL_FLAGS_LOCK:
        CANCEL_FLAGS.pop(session_id, None)


def check_cancelled(session_id: str) -> None:
    if not session_id:
        return
    with CANCEL_FLAGS_LOCK:
        cancelled = CANCEL_FLAGS.get(session_id, False)
    if cancelled:
        raise CancelledByUser("用户已取消本次操作。")


def normalize_caption(caption: str) -> str:
    normalized = caption.lower().strip()
    if normalized and not normalized.endswith("."):
        normalized += "."
    return normalized


def contains_chinese(text: str) -> bool:
    return bool(ZH_TEXT_PATTERN.search(text))


def translate_text_with_google(text: str) -> str:
    translator = GoogleTranslator(source="auto", target="en")
    translated = translator.translate(text)
    if not translated:
        raise PromptTranslationError("翻译服务暂时不可用，请稍后重试或直接输入英文。")
    return translated.strip()


def translate_prompt_if_needed(prompt: str, field_label: str) -> Tuple[str, Optional[str]]:
    raw_prompt = prompt.strip()
    if not raw_prompt:
        return raw_prompt, None

    if not contains_chinese(raw_prompt):
        return raw_prompt, None

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(translate_text_with_google, raw_prompt)
            translated_prompt = future.result(timeout=TRANSLATION_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        raise PromptTranslationTimeoutError("翻译服务超时，请直接输入英文。") from exc
    except (DeepTranslatorRequestError, TranslationNotFound, TooManyRequests) as exc:
        raise PromptTranslationError("翻译服务暂时不可用，请稍后重试或直接输入英文。") from exc
    except PromptTranslationError:
        raise
    except Exception as exc:
        raise PromptTranslationError("翻译服务暂时不可用，请稍后重试或直接输入英文。") from exc

    if not translated_prompt:
        raise PromptTranslationError("翻译服务暂时不可用，请稍后重试或直接输入英文。")

    note = f"{field_label}识别到中文输入，已自动转译为：[{translated_prompt}]"
    return translated_prompt, note


def prepare_detection_prompt(prompt: str, task_type: str) -> Tuple[str, Optional[str]]:
    raw_prompt = prompt.strip()
    if task_type == "automatic" or not raw_prompt:
        return raw_prompt, None
    return translate_prompt_if_needed(raw_prompt, "目标检测提示词")


def prepare_inpaint_prompt(prompt: str) -> Tuple[str, Optional[str]]:
    return translate_prompt_if_needed(prompt, "背景复原提示词")


def move_tensor_batch_to_device(batch: Dict[str, Any], device: str, dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def show_anns(anns: List[Dict[str, Any]]) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    if not anns:
        return None, None

    sorted_anns = sorted(anns, key=lambda item: item["area"], reverse=True)
    full_img = None
    encoded_map = None

    for index, ann in enumerate(sorted_anns):
        mask = ann["segmentation"]
        if full_img is None:
            full_img = np.zeros((mask.shape[0], mask.shape[1], 3))
            encoded_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint16)
        encoded_map[mask != 0] = index + 1
        full_img[mask != 0] = np.random.random((1, 3)).tolist()[0]

    full_img = Image.fromarray(np.uint8(full_img * 255))
    encoded = np.zeros((encoded_map.shape[0], encoded_map.shape[1], 3))
    encoded[:, :, 0] = encoded_map % 256
    encoded[:, :, 1] = encoded_map // 256
    return full_img, encoded


def draw_mask(mask: np.ndarray, draw: ImageDraw.ImageDraw, random_color: bool = False) -> None:
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            150,
        )
    else:
        color = (30, 144, 255, 150)

    for coord in np.transpose(np.nonzero(mask)):
        draw.point(coord[::-1], fill=color)


def draw_box(box: torch.Tensor, draw: ImageDraw.ImageDraw, label: str) -> None:
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=3)

    if not label:
        return

    font = ImageFont.load_default()
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((box[0], box[1]), label, font=font)
    else:
        width, height = draw.textsize(label, font=font)
        bbox = (box[0], box[1], box[0] + width, box[1] + height)
    draw.rectangle(bbox, fill=color)
    draw.text((box[0], box[1]), label, fill="white", font=font)


def transform_image(image_pil: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


def format_labels(labels: List[str], scores: torch.Tensor) -> List[str]:
    return [f"{label} ({score:.2f})" for label, score in zip(labels, scores.tolist())]


def build_status_markdown(
    task_type: str,
    backend_name: str,
    prompt_used: str,
    inpaint_prompt_used: Optional[str] = None,
    labels: Optional[List[str]] = None,
    note: Optional[str] = None,
    vram_mb: Optional[float] = None,
) -> str:
    lines = [
        "### 运行状态",
        f"- 任务模式: `{task_type}`",
        f"- 检测后端: `{backend_name}`",
        f"- 设备策略: `{DEVICE}` / `{MAIN_DTYPE}`",
    ]
    if prompt_used:
        lines.append(f"- 生效提示词: `{prompt_used}`")
    if inpaint_prompt_used:
        lines.append(f"- 背景复原提示词: `{inpaint_prompt_used}`")
    if labels:
        lines.append(f"- 检测结果: `{', '.join(labels[:8])}`")
    if vram_mb is not None:
        lines.append(f"- 峰值显存: `{vram_mb:.1f} MiB`")
    if note:
        lines.append(f"- 备注: {note}")
    return "\n".join(lines)


def build_error_markdown(message: str, detail: Optional[str] = None) -> str:
    lines = [
        "### 运行失败",
        f"- 错误原因: `{message}`",
    ]
    if detail:
        lines.append(f"- 详细信息: `{detail}`")
    lines.append("- 建议: 检查提示词、模型缓存与依赖版本，或查看终端日志中的完整 traceback。")
    return "\n".join(lines)


def build_idle_markdown() -> str:
    return "### 待命中\n- 当前没有正在执行的任务。\n- 你可以继续修改模式、提示词或图片后重新提交。"


def build_cancelled_markdown() -> str:
    return "### 操作已取消\n- 当前任务已停止等待，可立即切换模式或修改提示词后重新提交。"


def parse_image_editor_input(input_image: Any) -> Tuple[Image.Image, Optional[Image.Image]]:
    def to_pil_image(value: Any) -> Optional[Image.Image]:
        if value is None:
            return None
        if isinstance(value, Image.Image):
            return value
        return Image.fromarray(np.array(value))

    if isinstance(input_image, dict):
        if "image" in input_image or "mask" in input_image:
            image = input_image.get("image")
            mask = input_image.get("mask")
        else:
            background = to_pil_image(input_image.get("background"))
            composite = to_pil_image(input_image.get("composite"))
            layers = [to_pil_image(layer) for layer in input_image.get("layers", []) if layer is not None]
            image = background or composite
            mask = None
            if layers:
                mask = Image.new("L", layers[0].size, color=0)
                mask_np = np.zeros((layers[0].size[1], layers[0].size[0]), dtype=np.uint8)
                for layer in layers:
                    alpha = np.array(layer.convert("RGBA").getchannel("A"), dtype=np.uint8)
                    mask_np = np.maximum(mask_np, alpha)
                mask = Image.fromarray(mask_np, mode="L")
    else:
        image = input_image
        mask = None

    if image is None:
        raise ValueError("请先上传一张待处理图片。")

    image = to_pil_image(image)
    mask = to_pil_image(mask)

    return image.convert("RGB"), mask


def build_input_image_component() -> gr.components.Component:
    common_kwargs = {
        "type": "pil",
        "value": "assets/demo1.jpg",
        "label": "上传商品图片（支持直接在图上涂抹）",
    }

    if GRADIO_IMAGE_ACCEPTS_TOOL:
        image_kwargs = dict(common_kwargs)
        image_kwargs["tool"] = "sketch"
        if GRADIO_IMAGE_ACCEPTS_SOURCE:
            image_kwargs["source"] = "upload"
        else:
            image_kwargs["sources"] = "upload"
        return gr.Image(**image_kwargs)

    if hasattr(gr, "ImageEditor"):
        return gr.ImageEditor(
            **common_kwargs,
            sources="upload",
            brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
            eraser=gr.Eraser(),
            transforms=(),
            layers=True,
        )

    image_kwargs = dict(common_kwargs)
    if GRADIO_IMAGE_ACCEPTS_SOURCE:
        image_kwargs["source"] = "upload"
    else:
        image_kwargs["sources"] = "upload"
    return gr.Image(**image_kwargs)


def current_peak_vram_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 ** 2


class LocalGroundingBackend:
    name = "groundingdino-local"
    note = "本地 GroundingDINO 自定义算子 smoke test 通过"

    def __init__(self) -> None:
        self.model = None

    @staticmethod
    def custom_ops_available() -> bool:
        try:
            import groundingdino._C  # type: ignore

            return True
        except Exception:
            return False

    def load(self) -> None:
        if LOCAL_GROUNDING_IMPORT_ERROR is not None:
            raise RuntimeError(f"本地 GroundingDINO 导入失败: {LOCAL_GROUNDING_IMPORT_ERROR}")

        args = SLConfig.fromfile(CONFIG_FILE)
        args.device = DEVICE
        model = build_model(args)
        checkpoint = torch.load(LOCAL_GROUNDING_CHECKPOINT, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.to(device=DEVICE)
        if DEVICE == "cuda":
            model.half()
        self.model = model
        self._smoke_test()

    def _smoke_test(self) -> None:
        dummy = Image.new("RGB", (64, 64), color="white")
        image = transform_image(dummy).to(device=DEVICE, dtype=MAIN_DTYPE)
        with torch.inference_mode(), autocast_context():
            _ = self.model(image[None], captions=["object."])

    def predict(
        self,
        image_pil: Image.Image,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        caption = normalize_caption(caption)
        image = transform_image(image_pil).to(device=DEVICE, dtype=MAIN_DTYPE)

        with torch.inference_mode(), autocast_context():
            outputs = self.model(image[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        scores = []
        for logit, _ in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())
        return boxes_filt, torch.tensor(scores), pred_phrases

    def to(self, device: str) -> None:
        if self.model is not None:
            self.model.to(device=device)


class HFGroundingBackend:
    name = "groundingdino-hf"
    note = "使用 Hugging Face Grounding DINO，已禁用自定义 kernels"

    def __init__(self) -> None:
        self.processor = None
        self.model = None

    def load(self) -> None:
        processor_cls = GroundingDinoProcessor or AutoProcessor
        self.processor = processor_cls.from_pretrained(HF_GROUNDING_MODEL_ID)

        load_kwargs = {
            "torch_dtype": MAIN_DTYPE,
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                HF_GROUNDING_MODEL_ID,
                disable_custom_kernels=True,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                HF_GROUNDING_MODEL_ID,
                **load_kwargs,
            )
            if hasattr(self.model, "config") and hasattr(self.model.config, "disable_custom_kernels"):
                self.model.config.disable_custom_kernels = True

        self.model.eval()
        self.model.to(device=DEVICE)

    def predict(
        self,
        image_pil: Image.Image,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        caption = normalize_caption(caption)
        batch = self.processor(images=image_pil, text=caption, return_tensors="pt")
        batch = move_tensor_batch_to_device(batch, DEVICE, dtype=MAIN_DTYPE)

        with torch.inference_mode(), autocast_context():
            outputs = self.model(**batch)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            batch["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        result = results[0]
        boxes = result["boxes"].detach().cpu() if len(result["boxes"]) else torch.zeros((0, 4))
        scores = result["scores"].detach().cpu() if len(result["scores"]) else torch.zeros((0,))
        labels = list(result.get("text_labels", result["labels"]))
        return boxes, scores, labels

    def to(self, device: str) -> None:
        if self.model is not None:
            self.model.to(device=device)


def ensure_grounding_backend() -> Any:
    backend = MODEL_STATE["grounding_backend"]
    if backend is not None:
        return backend

    if LocalGroundingBackend.custom_ops_available():
        try:
            backend = LocalGroundingBackend()
            backend.load()
            MODEL_STATE["grounding_backend"] = backend
            return backend
        except Exception as exc:
            warnings.warn(f"本地 GroundingDINO backend 启动失败，切换到 HF fallback: {exc}")
            cleanup_cuda_memory()

    backend = HFGroundingBackend()
    backend.load()
    MODEL_STATE["grounding_backend"] = backend
    return backend


def reset_grounding_backend() -> None:
    backend = MODEL_STATE["grounding_backend"]
    if backend is not None and hasattr(backend, "to"):
        backend.to("cpu")
    MODEL_STATE["grounding_backend"] = None
    cleanup_cuda_memory()


def ensure_sam_components() -> Tuple[Any, SamPredictor, SamAutomaticMaskGenerator]:
    if MODEL_STATE["sam_predictor"] is not None:
        return (
            MODEL_STATE["sam_model"],
            MODEL_STATE["sam_predictor"],
            MODEL_STATE["sam_automask_generator"],
        )

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(f"未找到 SAM 权重: {SAM_CHECKPOINT}")

    sam_model = build_sam(checkpoint=SAM_CHECKPOINT)
    sam_model.to(device=DEVICE)

    predictor = SamPredictor(sam_model)
    automask_generator = SamAutomaticMaskGenerator(sam_model)
    MODEL_STATE["sam_model"] = sam_model
    MODEL_STATE["sam_predictor"] = predictor
    MODEL_STATE["sam_automask_generator"] = automask_generator
    return sam_model, predictor, automask_generator


def move_primary_models_to(device: str) -> None:
    backend = MODEL_STATE["grounding_backend"]
    if backend is not None:
        backend.to(device)

    sam_model = MODEL_STATE["sam_model"]
    if sam_model is not None:
        sam_model.to(device=device)


def ensure_primary_models_on_gpu() -> Tuple[Any, SamPredictor, SamAutomaticMaskGenerator]:
    ensure_cuda_available()
    backend = ensure_grounding_backend()
    _, predictor, automask_generator = ensure_sam_components()
    move_primary_models_to(DEVICE)
    return backend, predictor, automask_generator


def ensure_blip_components() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    processor = MODEL_STATE["blip_processor"]
    model = MODEL_STATE["blip_model"]
    if processor is None:
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        MODEL_STATE["blip_processor"] = processor
    if model is None:
        model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_ID,
            torch_dtype=BLIP_DTYPE,
        )
        model.eval()
        model.to(device=DEVICE)
        MODEL_STATE["blip_model"] = model
    return processor, model


def release_blip_model() -> None:
    model = MODEL_STATE["blip_model"]
    if model is not None:
        model.to("cpu")
    MODEL_STATE["blip_model"] = None
    cleanup_cuda_memory()


def generate_caption(raw_image: Image.Image) -> str:
    ensure_cuda_available()
    processor, blip_model = ensure_blip_components()
    batch = processor(images=raw_image, return_tensors="pt")
    batch = move_tensor_batch_to_device(batch, DEVICE, dtype=BLIP_DTYPE)

    with torch.inference_mode(), autocast_context():
        output = blip_model.generate(**batch, max_new_tokens=32)

    caption = processor.decode(output[0], skip_special_tokens=True)
    release_blip_model()
    return caption.strip()


def ensure_inpaint_pipeline() -> StableDiffusionInpaintPipeline:
    pipeline = MODEL_STATE["inpaint_pipeline"]
    if pipeline is not None:
        return pipeline

    ensure_cuda_available()
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=SD_DTYPE,
        low_cpu_mem_usage=True,
    )
    pipeline.enable_attention_slicing("auto")
    pipeline.enable_vae_slicing()
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if hasattr(pipeline, "requires_safety_checker"):
        pipeline.requires_safety_checker = False
    pipeline.enable_model_cpu_offload()
    MODEL_STATE["inpaint_pipeline"] = pipeline
    return pipeline


def run_grounding_detection(
    backend: Any,
    image_pil: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], str, str]:
    try:
        boxes_filt, scores, labels = backend.predict(image_pil, text_prompt, box_threshold, text_threshold)
        return boxes_filt, scores, labels, backend.name, backend.note
    except Exception as exc:
        if backend.name == "groundingdino-local":
            warnings.warn(f"本地 GroundingDINO 推理失败，自动回退到 HF backend: {exc}")
            reset_grounding_backend()
            fallback = ensure_grounding_backend()
            boxes_filt, scores, labels = fallback.predict(image_pil, text_prompt, box_threshold, text_threshold)
            return boxes_filt, scores, labels, fallback.name, fallback.note
        raise


def rescale_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int], normalized: bool) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    if not normalized:
        return boxes.float().cpu()

    height, width = size[1], size[0]
    scaled_boxes = boxes.clone()
    for index in range(scaled_boxes.size(0)):
        scaled_boxes[index] = scaled_boxes[index] * torch.tensor([width, height, width, height])
        scaled_boxes[index][:2] -= scaled_boxes[index][2:] / 2
        scaled_boxes[index][2:] += scaled_boxes[index][:2]
    return scaled_boxes.float().cpu()


def predict_masks_from_boxes(
    predictor: SamPredictor,
    image_np: np.ndarray,
    boxes_filt: torch.Tensor,
) -> torch.Tensor:
    predictor.set_image(image_np)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2]).to(DEVICE)
    with torch.inference_mode(), autocast_context():
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    return masks


def build_mask_outputs(size: Tuple[int, int], masks: torch.Tensor) -> Tuple[Image.Image, Image.Image]:
    overlay = Image.new("RGBA", size, color=(0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    binary_union = np.zeros((size[1], size[0]), dtype=np.uint8)

    for mask in masks:
        mask_np = mask[0].detach().cpu().numpy().astype(bool)
        draw_mask(mask_np, overlay_draw, random_color=True)
        binary_union = np.maximum(binary_union, mask_np.astype(np.uint8) * 255)

    mask_preview = Image.fromarray(binary_union, mode="L")
    return overlay, mask_preview


def render_segmentation_result(
    image_pil: Image.Image,
    boxes_filt: torch.Tensor,
    labels: List[str],
    masks: torch.Tensor,
    caption: Optional[str] = None,
) -> Tuple[Image.Image, Image.Image]:
    overlay, mask_preview = build_mask_outputs(image_pil.size, masks)
    annotated = image_pil.copy()
    draw = ImageDraw.Draw(annotated)
    for box, label in zip(boxes_filt, labels):
        draw_box(box, draw, label)
    if caption:
        draw.text((12, 12), caption, fill="black", font=ImageFont.load_default())

    composited = annotated.convert("RGBA")
    composited.alpha_composite(overlay)
    return composited, mask_preview


def run_scribble_task(
    image_pil: Image.Image,
    scribble: Optional[Image.Image],
    predictor: SamPredictor,
) -> Tuple[Image.Image, Image.Image]:
    if scribble is None:
        raise ValueError("涂抹模式需要在图片上用鼠标勾画目标区域。")

    image_np = np.array(image_pil)
    predictor.set_image(image_np)
    scribble_mask = np.array(scribble.convert("L")) > 0
    labeled_array, num_features = ndimage.label(scribble_mask)
    if num_features == 0:
        raise ValueError("请先在图片上画出至少一个涂抹点。")

    centers = ndimage.center_of_mass(scribble_mask, labeled_array, range(1, num_features + 1))
    point_coords = torch.tensor(centers, dtype=torch.float32)
    point_coords = predictor.transform.apply_coords_torch(point_coords, image_np.shape[:2]).unsqueeze(0).to(DEVICE)
    point_labels = torch.ones((1, len(centers)), dtype=torch.int64, device=DEVICE)

    with torch.inference_mode(), autocast_context():
        masks, _, _ = predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=None,
            boxes=None,
            multimask_output=False,
        )

    overlay, mask_preview = build_mask_outputs(image_pil.size, masks)
    composited = image_pil.convert("RGBA")
    composited.alpha_composite(overlay)
    return composited, mask_preview


def run_automask_task(image_pil: Image.Image, automask_generator: SamAutomaticMaskGenerator) -> Tuple[Image.Image, Optional[Image.Image]]:
    masks = automask_generator.generate(np.array(image_pil))
    result, _ = show_anns(masks)
    if result is None:
        raise ValueError("自动分割未生成任何掩码。")
    return result, None


def run_inpainting_task(
    image_pil: Image.Image,
    masks: torch.Tensor,
    inpaint_prompt: str,
    inpaint_mode: str,
    session_id: str,
) -> Tuple[Image.Image, Image.Image]:
    if not inpaint_prompt.strip():
        raise ValueError("背景复原模式需要填写“背景复原提示词”。")

    if inpaint_mode == "merge":
        combined_masks = torch.sum(masks, dim=0).unsqueeze(0)
        combined_masks = torch.where(combined_masks > 0, True, False)
    else:
        combined_masks = masks[:1]

    mask_np = combined_masks[0][0].detach().cpu().numpy().astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_np, mode="L")

    move_primary_models_to("cpu")
    cleanup_cuda_memory()
    pipeline = ensure_inpaint_pipeline()

    resized_image = image_pil.resize((MAX_INPAINT_SIZE, MAX_INPAINT_SIZE))
    resized_mask = mask_pil.resize((MAX_INPAINT_SIZE, MAX_INPAINT_SIZE), resample=Image.NEAREST)

    def cancel_callback(_pipeline, _step_index, _timestep, callback_kwargs):
        check_cancelled(session_id)
        return callback_kwargs

    with torch.inference_mode():
        output = pipeline(
            prompt=inpaint_prompt,
            image=resized_image,
            mask_image=resized_mask,
            num_inference_steps=DEFAULT_INPAINT_STEPS,
            callback_on_step_end=cancel_callback,
        ).images[0]

    cleanup_cuda_memory()
    return output.resize(image_pil.size), mask_pil


def run_grounded_sam(
    input_image: Any,
    text_prompt: str,
    task_type: str,
    inpaint_prompt: str,
    box_threshold: float,
    text_threshold: float,
    iou_threshold: float,
    inpaint_mode: str,
    scribble_mode: str,
    session_id: str,
) -> Tuple[Image.Image, Optional[Image.Image], str]:
    del scribble_mode
    ensure_cuda_available()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    set_cancel_flag(session_id, False)
    prompt_notes: List[str] = []

    try:
        image_pil, scribble = parse_image_editor_input(input_image)
        image_np = np.array(image_pil)
        check_cancelled(session_id)
        if task_type == "automask":
            _, _, automask_generator = ensure_primary_models_on_gpu()
            check_cancelled(session_id)
            result_image, mask_preview = run_automask_task(image_pil, automask_generator)
            check_cancelled(session_id)
            status = build_status_markdown(
                task_type,
                "sam-automask",
                "",
                note="整图自动分割不依赖 GroundingDINO。",
                vram_mb=current_peak_vram_mb(),
            )
            return result_image, mask_preview, status

        if task_type == "scribble":
            _, predictor, _ = ensure_primary_models_on_gpu()
            check_cancelled(session_id)
            result_image, mask_preview = run_scribble_task(image_pil, scribble, predictor)
            check_cancelled(session_id)
            status = build_status_markdown(
                task_type,
                "sam-scribble",
                "",
                note="已基于你的手绘点位完成交互式分割。",
                vram_mb=current_peak_vram_mb(),
            )
            return result_image, mask_preview, status

        backend, predictor, _ = ensure_primary_models_on_gpu()
        prompt_used = text_prompt.strip()
        inpaint_prompt_used = inpaint_prompt.strip()
        if task_type == "automatic":
            prompt_used = generate_caption(image_pil)
        else:
            prompt_used, prompt_note = prepare_detection_prompt(prompt_used, task_type)
            if prompt_note:
                prompt_notes.append(prompt_note)
        if not prompt_used:
            raise ValueError("请填写“商品目标描述”，或者切换到 automatic 自动识别模式。")

        if task_type == "inpainting":
            inpaint_prompt_used, inpaint_prompt_note = prepare_inpaint_prompt(inpaint_prompt_used)
            if inpaint_prompt_note:
                prompt_notes.append(inpaint_prompt_note)

        check_cancelled(session_id)
        boxes_filt, scores, raw_labels, backend_name, backend_note = run_grounding_detection(
            backend,
            image_pil,
            prompt_used,
            box_threshold,
            text_threshold,
        )
        check_cancelled(session_id)

        normalized_boxes = backend_name == "groundingdino-local"
        boxes_filt = rescale_boxes_to_image(boxes_filt, image_pil.size, normalized=normalized_boxes)
        if boxes_filt.size(0) == 0:
            status = build_status_markdown(
                task_type,
                backend_name,
                prompt_used,
                note="；".join(part for part in [backend_note, *prompt_notes, "当前阈值下未检测到目标。"] if part),
                vram_mb=current_peak_vram_mb(),
            )
            return image_pil, None, status

        if task_type == "automatic":
            keep = torchvision.ops.nms(boxes_filt, scores, iou_threshold)
            boxes_filt = boxes_filt[keep]
            scores = scores[keep]
            raw_labels = [raw_labels[index] for index in keep.tolist()]

        formatted_labels = format_labels(raw_labels, scores)

        if task_type == "det":
            annotated = image_pil.copy()
            draw = ImageDraw.Draw(annotated)
            for box, label in zip(boxes_filt, formatted_labels):
                draw_box(box, draw, label)
            status = build_status_markdown(
                task_type,
                backend_name,
                prompt_used,
                labels=formatted_labels,
                note="；".join(part for part in [backend_note, *prompt_notes] if part),
                vram_mb=current_peak_vram_mb(),
            )
            return annotated, None, status

        check_cancelled(session_id)
        masks = predict_masks_from_boxes(predictor, image_np, boxes_filt)
        check_cancelled(session_id)
        if task_type in {"seg", "automatic"}:
            result_image, mask_preview = render_segmentation_result(
                image_pil,
                boxes_filt,
                formatted_labels,
                masks,
                caption=prompt_used if task_type == "automatic" else None,
            )
            status = build_status_markdown(
                task_type,
                backend_name,
                prompt_used,
                labels=formatted_labels,
                note="；".join(part for part in [backend_note, *prompt_notes] if part),
                vram_mb=current_peak_vram_mb(),
            )
            return result_image, mask_preview, status

        if task_type == "inpainting":
            result_image, mask_preview = run_inpainting_task(
                image_pil=image_pil,
                masks=masks,
                inpaint_prompt=inpaint_prompt_used,
                inpaint_mode=inpaint_mode,
                session_id=session_id,
            )
            check_cancelled(session_id)
            status = build_status_markdown(
                task_type,
                backend_name,
                prompt_used,
                inpaint_prompt_used=inpaint_prompt_used,
                labels=formatted_labels,
                note="；".join(part for part in [backend_note, *prompt_notes, "Stable Diffusion 已开启 CPU offload 与 attention slicing。"] if part),
                vram_mb=current_peak_vram_mb(),
            )
            return result_image, mask_preview, status

        raise ValueError(f"不支持的任务模式: {task_type}")
    except CancelledByUser:
        cleanup_cuda_memory()
        return None, None, build_cancelled_markdown()
    except Exception:
        cleanup_cuda_memory()
        raise
    finally:
        clear_cancel_flag(session_id)


def run_grounded_sam_ui(
    input_image: Any,
    text_prompt: str,
    task_type: str,
    inpaint_prompt: str,
    box_threshold: float,
    text_threshold: float,
    iou_threshold: float,
    inpaint_mode: str,
    scribble_mode: str,
    session_id: str,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    try:
        return run_grounded_sam(
            input_image=input_image,
            text_prompt=text_prompt,
            task_type=task_type,
            inpaint_prompt=inpaint_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            iou_threshold=iou_threshold,
            inpaint_mode=inpaint_mode,
            scribble_mode=scribble_mode,
            session_id=session_id,
        )
    except PromptTranslationTimeoutError as exc:
        fallback_image = None
        try:
            fallback_image, _ = parse_image_editor_input(input_image)
        except Exception:
            fallback_image = None
        return fallback_image, None, build_error_markdown(message=str(exc))
    except PromptTranslationError as exc:
        fallback_image = None
        try:
            fallback_image, _ = parse_image_editor_input(input_image)
        except Exception:
            fallback_image = None
        return fallback_image, None, build_error_markdown(message=str(exc))
    except Exception as exc:
        traceback.print_exc()
        fallback_image = None
        try:
            fallback_image, _ = parse_image_editor_input(input_image)
        except Exception:
            fallback_image = None
        message = f"{exc.__class__.__name__}: {exc}"
        return fallback_image, None, build_error_markdown(
            message=message,
            detail="错误已透传到界面状态区，同时完整 traceback 已写入终端日志。",
        )


def cancel_current_run(session_id: str) -> Tuple[None, None, str]:
    set_cancel_flag(session_id, True)
    return None, None, build_cancelled_markdown()


def build_mode_hint(task_type: str) -> str:
    hints = {
        "seg": "主流程推荐。输入商品主体描述，输出检测框叠加后的精细分割结果。",
        "inpainting": "面向背景复原。先锁定目标，再用 Stable Diffusion 对目标区域重绘或替换。",
        "det": "只做目标检测，适合快速校准提示词是否命中。",
        "automatic": "自动识别模式。使用 BLIP 先生成 caption，再联动 GroundingDINO + SAM。",
        "scribble": "交互式涂抹分割。直接在图片上画点，无需文字提示词。",
        "automask": "整图一次性分割全部区域，适合素材预分析。",
    }
    return f"**模式说明**：{hints.get(task_type, '')}"


def build_app() -> gr.Blocks:
    block_kwargs = {"title": "基于多模态大模型的特定目标提取与背景复原"}
    if GRADIO_BLOCKS_ACCEPTS_CSS:
        block_kwargs["css"] = APP_CSS
    block = gr.Blocks(**block_kwargs)

    with block:
        gr.HTML(
            """
            <div class="hero-card">
                <span class="hero-badge">GPU SaaS Console</span>
                <h1 class="hero-title">基于多模态大模型的特定目标提取与背景复原</h1>
                <p class="hero-text">
                    面向电商主图、白底图、营销素材与旧图翻新场景，统一接入 GroundingDINO、SAM 与 Stable Diffusion。
                    左侧完成上传与策略配置，右侧直接查看大图结果与掩码，保证项目演示时既有技术可信度，也有产品质感。
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=380):
                session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
                gr.HTML(
                    """
                    <div class="panel-card">
                        <h2 class="panel-title">智能处理台</h2>
                        <p class="panel-text">
                            默认面向“商品主体提取”和“背景复原”两条主流程设计，其余模式也统一走同一套 GPU 推理与显存保护策略。
                        </p>
                    </div>
                    """
                )
                input_image = build_input_image_component()
                task_type = gr.Dropdown(
                    choices=[
                        ("目标提取（分割）", "seg"),
                        ("背景复原（重绘）", "inpainting"),
                        ("目标检测", "det"),
                        ("自动识别", "automatic"),
                        ("交互式涂抹分割", "scribble"),
                        ("整图自动分割", "automask"),
                    ],
                    value="seg",
                    label="处理模式",
                    info="推荐先用“目标提取（分割）”验证检测与分割链路。",
                )
                mode_hint = gr.Markdown(build_mode_hint("seg"))
                text_prompt = gr.Textbox(
                    label="商品目标描述",
                    placeholder="例如：白色马克杯 / 模特手中的包 / 画面中的支架",
                    lines=2,
                )
                inpaint_prompt = gr.Textbox(
                    label="背景复原提示词",
                    placeholder="例如：高质感大理石台面，柔和棚拍光线，适合电商主图",
                    lines=2,
                )
                with gr.Row():
                    run_button = gr.Button("一键处理", elem_id="run-btn", variant="primary")
                    stop_button = gr.Button("停止处理", elem_id="stop-btn", variant="stop")

                with gr.Accordion("高级参数", open=False):
                    box_threshold = gr.Slider(
                        label="检测阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                    )
                    text_threshold = gr.Slider(
                        label="文本阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                    )
                    iou_threshold = gr.Slider(
                        label="NMS IOU 阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )
                    inpaint_mode = gr.Dropdown(
                        choices=[("合并全部目标区域", "merge"), ("仅处理首个目标", "first")],
                        value="merge",
                        label="背景复原掩码策略",
                    )
                    scribble_mode = gr.Dropdown(
                        choices=[("单一目标", "merge"), ("按点拆分", "split")],
                        value="split",
                        label="涂抹点分组策略",
                    )

            with gr.Column(scale=2, min_width=720):
                gr.HTML(
                    """
                    <div class="panel-card">
                        <h2 class="panel-title">处理结果总览</h2>
                        <p class="panel-text">
                            右侧优先展示最终大图结果，下面提供掩码辅助图和运行状态，便于演示时解释检测后端、显存策略与推理路径。
                        </p>
                    </div>
                    """
                )
                result_image = gr.Image(
                    type="pil",
                    label="最终处理结果",
                    elem_id="result-image",
                    height=640,
                )
                mask_image = gr.Image(
                    type="pil",
                    label="辅助掩码 / 参考图",
                    elem_id="mask-image",
                    height=300,
                )
                status_box = gr.Markdown(
                    value=build_idle_markdown(),
                    elem_classes="status-card",
                )

        task_type.change(fn=build_mode_hint, inputs=task_type, outputs=mode_hint)
        run_event = run_button.click(
            fn=run_grounded_sam_ui,
            inputs=[
                input_image,
                text_prompt,
                task_type,
                inpaint_prompt,
                box_threshold,
                text_threshold,
                iou_threshold,
                inpaint_mode,
                scribble_mode,
                session_id_state,
            ],
            outputs=[result_image, mask_image, status_box],
        )
        stop_button.click(
            fn=cancel_current_run,
            inputs=[session_id_state],
            outputs=[result_image, mask_image, status_box],
            cancels=[run_event],
            queue=False,
        )

    if GRADIO_QUEUE_ACCEPTS_CONCURRENCY_COUNT:
        return block.queue(concurrency_count=MAX_GRADIO_CONCURRENCY, max_size=4)
    return block.queue(default_concurrency_limit=MAX_GRADIO_CONCURRENCY, max_size=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM GPU demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="使用 Gradio debug 模式")
    parser.add_argument("--share", action="store_true", help="开启外部分享")
    parser.add_argument("--port", type=int, default=7589, help="服务端口")
    args = parser.parse_args()

    validate_runtime_environment()
    print(args)
    print(f"Python device target: {DEVICE}")
    print(f"Main dtype: {MAIN_DTYPE}")
    print(f"Grounding fallback model: {HF_GROUNDING_MODEL_ID}")
    print(f"BLIP model: {BLIP_MODEL_ID}")
    print(f"Inpaint model: {INPAINT_MODEL_ID}")

    app = build_app()
    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": args.port,
        "debug": args.debug,
        "share": args.share,
    }
    if GRADIO_LAUNCH_ACCEPTS_CSS and not GRADIO_BLOCKS_ACCEPTS_CSS:
        launch_kwargs["css"] = APP_CSS
    app.launch(**launch_kwargs)
