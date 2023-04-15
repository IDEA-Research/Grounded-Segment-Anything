import os
import os.path as osp
import sys

class Config:

    ## model setting
    resnet_type = 50
    hand_resnet_type = 50
    face_resnet_type = 18

    model_type = 'OSX'

    # osx_model_settiing
    upscale = 4
    with_wrist = False
    hand_pos_joint_num = 20
    num_task_token = 24
    feat_dim = 1024
    encoder_config_file = 'osx/transformer_utils/configs/osx/encoder/body_encoder_large.py'

    ## input, output
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    output_hm_shape = (16, 16, 12)
    # output_hm_shape = (8, 8, 6)
    input_hand_shape = (256, 256)
    # output_hand_hm_shape = (8, 8, 8)
    output_hand_hm_shape = (16, 16, 16)
    input_face_shape = (192, 192)
    focal = (5000, 5000) # virtual focal lengths
    princpt = (input_body_shape[1]/2, input_body_shape[0]/2) # virtual principal point position
    body_3d_size = 2
    hand_3d_size = 0.3
    face_3d_size = 0.3
    camera_3d_size = 2.5

    ## human models
    flame_shape_params = 100
    flame_expression_params = 50

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    # root_dir = osp.join(cur_dir, '..')

    human_model_path = osp.join(cur_dir, 'utils', 'human_model_files')

cfg = Config()
