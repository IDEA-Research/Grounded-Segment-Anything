CUDA_VISIBLE_DEVICES=6 python demo.py \
  -c GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p groundingdino_swint_ogc.pth \
  -i assets/demo1.jpg \
  -o "outputs/0" \
  -t "bear" \