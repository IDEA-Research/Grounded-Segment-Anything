dataset_info = dict(
    dataset_name='panoptic_pose_3d',
    paper_info=dict(
        author='Joo, Hanbyul and Simon, Tomas and  Li, Xulong'
        'and Liu, Hao and Tan, Lei and Gui, Lin and Banerjee, Sean'
        'and Godisart, Timothy and Nabbe, Bart and Matthews, Iain'
        'and Kanade, Takeo and Nobuhara, Shohei and Sheikh, Yaser',
        title='Panoptic Studio: A Massively Multiview System '
        'for Interaction Motion Capture',
        container='IEEE Transactions on Pattern Analysis'
        ' and Machine Intelligence',
        year='2017',
        homepage='http://domedb.perception.cs.cmu.edu',
    ),
    keypoint_info={
        0:
        dict(name='neck', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(name='nose', id=1, color=[51, 153, 255], type='upper', swap=''),
        2:
        dict(name='mid_hip', id=2, color=[0, 255, 0], type='lower', swap=''),
        3:
        dict(
            name='left_shoulder',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        4:
        dict(
            name='left_elbow',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        5:
        dict(
            name='left_wrist',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='left_knee',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        8:
        dict(
            name='left_ankle',
            id=8,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        9:
        dict(
            name='right_shoulder',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        10:
        dict(
            name='right_elbow',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        11:
        dict(
            name='right_wrist',
            id=11,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='right_knee',
            id=13,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        14:
        dict(
            name='right_ankle',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        15:
        dict(
            name='left_eye',
            id=15,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        16:
        dict(
            name='left_ear',
            id=16,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        17:
        dict(
            name='right_eye',
            id=17,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        18:
        dict(
            name='right_ear',
            id=18,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear')
    },
    skeleton_info={
        0: dict(link=('nose', 'neck'), id=0, color=[51, 153, 255]),
        1: dict(link=('neck', 'left_shoulder'), id=1, color=[0, 255, 0]),
        2: dict(link=('neck', 'right_shoulder'), id=2, color=[255, 128, 0]),
        3: dict(link=('left_shoulder', 'left_elbow'), id=3, color=[0, 255, 0]),
        4: dict(
            link=('right_shoulder', 'right_elbow'), id=4, color=[255, 128, 0]),
        5: dict(link=('left_elbow', 'left_wrist'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('right_elbow', 'right_wrist'), id=6, color=[255, 128, 0]),
        7: dict(link=('left_ankle', 'left_knee'), id=7, color=[0, 255, 0]),
        8: dict(link=('left_knee', 'left_hip'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_ankle', 'right_knee'), id=9, color=[255, 128, 0]),
        10: dict(link=('right_knee', 'right_hip'), id=10, color=[255, 128, 0]),
        11: dict(link=('mid_hip', 'left_hip'), id=11, color=[0, 255, 0]),
        12: dict(link=('mid_hip', 'right_hip'), id=12, color=[255, 128, 0]),
        13: dict(link=('mid_hip', 'neck'), id=13, color=[51, 153, 255]),
    },
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.0, 1.2, 1.5, 1.0, 1.2, 1.5, 1.0, 1.2,
        1.5, 1.0, 1.0, 1.0, 1.0
    ],
    sigmas=[
        0.026, 0.026, 0.107, 0.079, 0.072, 0.062, 0.107, 0.087, 0.089, 0.079,
        0.072, 0.062, 0.107, 0.087, 0.089, 0.025, 0.035, 0.025, 0.035
    ])
