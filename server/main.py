import sys
import os
import time
import json # config 파싱 및 오류 처리용

import cv2
import numpy as np

# FastAPI 모듈 임포트. JSONResponse 대신 Response를 사용합니다.
from fastapi import FastAPI, UploadFile, File, Form, Response 
from fastapi.responses import JSONResponse 

import supervision as sv
import torch
import torchvision

# --- ⭐ 1단계: 경로 문제 해결을 위한 sys.path 설정 ⭐ ---
# Uvicorn 환경에서 모듈 임포트 문제를 해결하기 위해 프로젝트 루트와 EfficientSAM을 경로에 추가합니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # ~/Workspace/Grounded-Segment-Anything

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EFFICIENT_SAM_ROOT = os.path.join(PROJECT_ROOT, "EfficientSAM")
if EFFICIENT_SAM_ROOT not in sys.path:
    sys.path.insert(1, EFFICIENT_SAM_ROOT) 
# --- ⭐ 경로 설정 완료 ⭐ ---


from groundingdino.util.inference import Model
# segment_anything 폴더명을 변경했다면 'segment_anything_repo'로 바꿔야 합니다. 
from segment_anything import SamPredictor 
# sys.path에 EfficientSAM을 추가했으므로, LightHQSAM부터 임포트합니다.
from LightHQSAM.setup_light_hqsam import setup_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
BASE_PATH = PROJECT_ROOT # BASE_PATH를 절대 경로로 수정하여 유연성 확보
GROUNDING_DINO_CONFIG_PATH = os.path.join(BASE_PATH,"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(BASE_PATH,"groundingdino_swint_ogc.pth")

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

# Building MobileSAM predictor
HQSAM_CHECKPOINT_PATH = os.path.join(BASE_PATH,"EfficientSAM/sam_hq_vit_tiny.pth")
checkpoint = torch.load(HQSAM_CHECKPOINT_PATH, map_location=DEVICE)
light_hqsam = setup_model()
light_hqsam.load_state_dict(checkpoint, strict=True)
light_hqsam.to(device=DEVICE)

sam_predictor = SamPredictor(light_hqsam)

IMG_SAVE_PATH = os.path.join(PROJECT_ROOT, "recv_images/") # 절대 경로 사용
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# Prompting SAM with detected boxes (함수 정의는 외부에 위치)
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False,
            hq_token_only=True,
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


@app.post("/infer")
async def infer(
    ts: str = Form(...),             # Timestamp
    image: UploadFile = File(...),   # Image File
    requester_id: str = Form(...),   # requester ID
    requester_pw: str = Form(...),   # Requester Password
    config: str = Form(default="")     # Configuration as JSON string
):
    # 1) 파일 내용 읽기 (bytes)
    content = await image.read()

    # 2) bytes → numpy → OpenCV 이미지(BGR)
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "Failed to decode image"},
        )

    server_ts = int(time.time() * 1000)
    SAVE_PATH = os.path.join(IMG_SAVE_PATH,f"received_image_{server_ts}.jpg")
    cv2.imwrite(SAVE_PATH, img)

    # --- ⭐ 2단계: config JSON 파싱 및 변수 추출 ⭐ ---
    try:
        parsed_config = json.loads(config)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid config JSON format"},
        )
        
    CLASSES = parsed_config.get('classes', [])
    BOX_THRESHOLD = parsed_config.get('box_threshold', 0.25)
    TEXT_THRESHOLD = parsed_config.get('text_threshold', 0.25)
    NMS_THRESHOLD = parsed_config.get('nms_threshold', 0.8)
    # --- ⭐ 파싱 완료 ⭐ ---
    
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=img,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    
    if len(detections.xyxy) > 0:
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # --- ⭐ 3단계: 첫 번째 마스크 PNG 바이너리 반환 ⭐ ---
    if detections.mask is None or len(detections.mask) == 0:
        # 감지된 객체가 없으면 204 No Content 응답
        return Response(
            content=json.dumps({"detail": "No objects detected for segmentation."}),
            status_code=204,
            media_type="application/json"
        )
    
    # 첫 번째 마스크만 선택 (bool 배열)
    first_mask_bool = detections.mask[0]

    # bool → uint8 (0 또는 255)로 변환
    mask_uint8_255 = (first_mask_bool.astype(np.uint8) * 255)

    # PNG 형식으로 메모리 버퍼에 인코딩 (무손실 압축)
    is_success, buffer = cv2.imencode(".png", mask_uint8_255)

    if not is_success:
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to encode mask to PNG."}
        )

    # FastAPI Response 객체를 사용하여 바이너리 데이터 반환
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        # 메타데이터는 헤더에 포함하여 보낼 수 있습니다 (선택 사항)
        headers={
            "X-Detected-Label": f"{CLASSES[detections.class_id[0]]}",
            "X-Confidence": f"{detections.confidence[0]:0.2f}",
            "X-Timestamp": ts 
        }
    )
    # --- ⭐ 반환 완료 ⭐ ---


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)