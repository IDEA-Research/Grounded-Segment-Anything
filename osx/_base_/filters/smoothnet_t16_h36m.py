# Config for SmoothNet filter trained on Human3.6M data with a window size of
# 16. The model is trained using root-centered keypoint coordinates around the
# pelvis (index:0), thus we set root_index=0 for the filter
filter_cfg = dict(
    type='SmoothNetFilter',
    window_size=16,
    output_size=16,
    checkpoint='https://download.openmmlab.com/mmpose/plugin/smoothnet/'
    'smoothnet_ws16_h36m.pth',
    hidden_size=512,
    res_hidden_size=256,
    num_blocks=3,
    root_index=0)
