dataset_info = dict(
    dataset_name='mpii_trb',
    paper_info=dict(
        author='Duan, Haodong and Lin, Kwan-Yee and Jin, Sheng and '
        'Liu, Wentao and Qian, Chen and Ouyang, Wanli',
        title='TRB: A Novel Triplet Representation for '
        'Understanding 2D Human Body',
        container='Proceedings of the IEEE International '
        'Conference on Computer Vision',
        year='2019',
        homepage='https://github.com/kennymckormick/'
        'Triplet-Representation-of-human-Body',
    ),
    keypoint_info={
        0:
        dict(
            name='left_shoulder',
            id=0,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='right_shoulder',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_wrist',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='left_knee',
            id=8,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_ankle',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        12:
        dict(name='head', id=12, color=[51, 153, 255], type='upper', swap=''),
        13:
        dict(name='neck', id=13, color=[51, 153, 255], type='upper', swap=''),
        14:
        dict(
            name='right_neck',
            id=14,
            color=[255, 255, 255],
            type='upper',
            swap='left_neck'),
        15:
        dict(
            name='left_neck',
            id=15,
            color=[255, 255, 255],
            type='upper',
            swap='right_neck'),
        16:
        dict(
            name='medial_right_shoulder',
            id=16,
            color=[255, 255, 255],
            type='upper',
            swap='medial_left_shoulder'),
        17:
        dict(
            name='lateral_right_shoulder',
            id=17,
            color=[255, 255, 255],
            type='upper',
            swap='lateral_left_shoulder'),
        18:
        dict(
            name='medial_right_bow',
            id=18,
            color=[255, 255, 255],
            type='upper',
            swap='medial_left_bow'),
        19:
        dict(
            name='lateral_right_bow',
            id=19,
            color=[255, 255, 255],
            type='upper',
            swap='lateral_left_bow'),
        20:
        dict(
            name='medial_right_wrist',
            id=20,
            color=[255, 255, 255],
            type='upper',
            swap='medial_left_wrist'),
        21:
        dict(
            name='lateral_right_wrist',
            id=21,
            color=[255, 255, 255],
            type='upper',
            swap='lateral_left_wrist'),
        22:
        dict(
            name='medial_left_shoulder',
            id=22,
            color=[255, 255, 255],
            type='upper',
            swap='medial_right_shoulder'),
        23:
        dict(
            name='lateral_left_shoulder',
            id=23,
            color=[255, 255, 255],
            type='upper',
            swap='lateral_right_shoulder'),
        24:
        dict(
            name='medial_left_bow',
            id=24,
            color=[255, 255, 255],
            type='upper',
            swap='medial_right_bow'),
        25:
        dict(
            name='lateral_left_bow',
            id=25,
            color=[255, 255, 255],
            type='upper',
            swap='lateral_right_bow'),
        26:
        dict(
            name='medial_left_wrist',
            id=26,
            color=[255, 255, 255],
            type='upper',
            swap='medial_right_wrist'),
        27:
        dict(
            name='lateral_left_wrist',
            id=27,
            color=[255, 255, 255],
            type='upper',
            swap='lateral_right_wrist'),
        28:
        dict(
            name='medial_right_hip',
            id=28,
            color=[255, 255, 255],
            type='lower',
            swap='medial_left_hip'),
        29:
        dict(
            name='lateral_right_hip',
            id=29,
            color=[255, 255, 255],
            type='lower',
            swap='lateral_left_hip'),
        30:
        dict(
            name='medial_right_knee',
            id=30,
            color=[255, 255, 255],
            type='lower',
            swap='medial_left_knee'),
        31:
        dict(
            name='lateral_right_knee',
            id=31,
            color=[255, 255, 255],
            type='lower',
            swap='lateral_left_knee'),
        32:
        dict(
            name='medial_right_ankle',
            id=32,
            color=[255, 255, 255],
            type='lower',
            swap='medial_left_ankle'),
        33:
        dict(
            name='lateral_right_ankle',
            id=33,
            color=[255, 255, 255],
            type='lower',
            swap='lateral_left_ankle'),
        34:
        dict(
            name='medial_left_hip',
            id=34,
            color=[255, 255, 255],
            type='lower',
            swap='medial_right_hip'),
        35:
        dict(
            name='lateral_left_hip',
            id=35,
            color=[255, 255, 255],
            type='lower',
            swap='lateral_right_hip'),
        36:
        dict(
            name='medial_left_knee',
            id=36,
            color=[255, 255, 255],
            type='lower',
            swap='medial_right_knee'),
        37:
        dict(
            name='lateral_left_knee',
            id=37,
            color=[255, 255, 255],
            type='lower',
            swap='lateral_right_knee'),
        38:
        dict(
            name='medial_left_ankle',
            id=38,
            color=[255, 255, 255],
            type='lower',
            swap='medial_right_ankle'),
        39:
        dict(
            name='lateral_left_ankle',
            id=39,
            color=[255, 255, 255],
            type='lower',
            swap='lateral_right_ankle'),
    },
    skeleton_info={
        0:
        dict(link=('head', 'neck'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('neck', 'left_shoulder'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('neck', 'right_shoulder'), id=2, color=[51, 153, 255]),
        3:
        dict(link=('left_shoulder', 'left_elbow'), id=3, color=[0, 255, 0]),
        4:
        dict(
            link=('right_shoulder', 'right_elbow'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('left_elbow', 'left_wrist'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('right_elbow', 'right_wrist'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('left_shoulder', 'left_hip'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('right_shoulder', 'right_hip'), id=8, color=[51, 153, 255]),
        9:
        dict(link=('left_hip', 'right_hip'), id=9, color=[51, 153, 255]),
        10:
        dict(link=('left_hip', 'left_knee'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_hip', 'right_knee'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_knee', 'left_ankle'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('right_knee', 'right_ankle'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('right_neck', 'left_neck'), id=14, color=[255, 255, 255]),
        15:
        dict(
            link=('medial_right_shoulder', 'lateral_right_shoulder'),
            id=15,
            color=[255, 255, 255]),
        16:
        dict(
            link=('medial_right_bow', 'lateral_right_bow'),
            id=16,
            color=[255, 255, 255]),
        17:
        dict(
            link=('medial_right_wrist', 'lateral_right_wrist'),
            id=17,
            color=[255, 255, 255]),
        18:
        dict(
            link=('medial_left_shoulder', 'lateral_left_shoulder'),
            id=18,
            color=[255, 255, 255]),
        19:
        dict(
            link=('medial_left_bow', 'lateral_left_bow'),
            id=19,
            color=[255, 255, 255]),
        20:
        dict(
            link=('medial_left_wrist', 'lateral_left_wrist'),
            id=20,
            color=[255, 255, 255]),
        21:
        dict(
            link=('medial_right_hip', 'lateral_right_hip'),
            id=21,
            color=[255, 255, 255]),
        22:
        dict(
            link=('medial_right_knee', 'lateral_right_knee'),
            id=22,
            color=[255, 255, 255]),
        23:
        dict(
            link=('medial_right_ankle', 'lateral_right_ankle'),
            id=23,
            color=[255, 255, 255]),
        24:
        dict(
            link=('medial_left_hip', 'lateral_left_hip'),
            id=24,
            color=[255, 255, 255]),
        25:
        dict(
            link=('medial_left_knee', 'lateral_left_knee'),
            id=25,
            color=[255, 255, 255]),
        26:
        dict(
            link=('medial_left_ankle', 'lateral_left_ankle'),
            id=26,
            color=[255, 255, 255])
    },
    joint_weights=[1.] * 40,
    sigmas=[])
