dataset_info = dict(
    dataset_name='aic',
    paper_info=dict(
        author='Wu, Jiahong and Zheng, He and Zhao, Bo and '
        'Li, Yixin and Yan, Baoming and Liang, Rui and '
        'Wang, Wenjia and Zhou, Shipei and Lin, Guosen and '
        'Fu, Yanwei and others',
        title='Ai challenger: A large-scale dataset for going '
        'deeper in image understanding',
        container='arXiv',
        year='2017',
        homepage='https://github.com/AIChallenger/AI_Challenger_2017',
    ),
    keypoint_info={
        0:
        dict(
            name='right_shoulder',
            id=0,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        1:
        dict(
            name='right_elbow',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        2:
        dict(
            name='right_wrist',
            id=2,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
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
            name='right_hip',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        7:
        dict(
            name='right_knee',
            id=7,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        8:
        dict(
            name='right_ankle',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        9:
        dict(
            name='left_hip',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        10:
        dict(
            name='left_knee',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        11:
        dict(
            name='left_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        12:
        dict(
            name='head_top',
            id=12,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        13:
        dict(name='neck', id=13, color=[51, 153, 255], type='upper', swap='')
    },
    skeleton_info={
        0:
        dict(link=('right_wrist', 'right_elbow'), id=0, color=[255, 128, 0]),
        1: dict(
            link=('right_elbow', 'right_shoulder'), id=1, color=[255, 128, 0]),
        2: dict(link=('right_shoulder', 'neck'), id=2, color=[51, 153, 255]),
        3: dict(link=('neck', 'left_shoulder'), id=3, color=[51, 153, 255]),
        4: dict(link=('left_shoulder', 'left_elbow'), id=4, color=[0, 255, 0]),
        5: dict(link=('left_elbow', 'left_wrist'), id=5, color=[0, 255, 0]),
        6: dict(link=('right_ankle', 'right_knee'), id=6, color=[255, 128, 0]),
        7: dict(link=('right_knee', 'right_hip'), id=7, color=[255, 128, 0]),
        8: dict(link=('right_hip', 'left_hip'), id=8, color=[51, 153, 255]),
        9: dict(link=('left_hip', 'left_knee'), id=9, color=[0, 255, 0]),
        10: dict(link=('left_knee', 'left_ankle'), id=10, color=[0, 255, 0]),
        11: dict(link=('head_top', 'neck'), id=11, color=[51, 153, 255]),
        12: dict(
            link=('right_shoulder', 'right_hip'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('left_shoulder', 'left_hip'), id=13, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.
    ],

    # 'https://github.com/AIChallenger/AI_Challenger_2017/blob/master/'
    # 'Evaluation/keypoint_eval/keypoint_eval.py#L50'
    # delta = 2 x sigma
    sigmas=[
        0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144,
        0.03909642, 0.03686941, 0.01981803, 0.03843971, 0.03412318, 0.02415081,
        0.01291456, 0.01236173
    ])
