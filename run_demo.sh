CUDA_VISIBLE_DEVICES=6 python demo.py \
  -c GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p groundingdino_swint_ogc.pth \
  -i ./cats.png \
  -o "outputs/0" \
  -t "the cat" \