dataset_info = dict(
    dataset_name='mhp',
    paper_info=dict(
        author='Zhao, Jian and Li, Jianshu and Cheng, Yu and '
        'Sim, Terence and Yan, Shuicheng and Feng, Jiashi',
        title='Understanding humans in crowded scenes: '
        'Deep nested adversarial learning and a '
        'new benchmark for multi-human parsing',
        container='Proceedings of the 26th ACM '
        'international conference on Multimedia',
        year='2018',
        homepage='https://lv-mhp.github.io/dataset',
    ),
    keypoint_info={
        0:
        dict(
            name='right_ankle',
            id=0,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        1:
        dict(
            name='right_knee',
            id=1,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        2:
        dict(
            name='right_hip',
            id=2,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        3:
        dict(
            name='left_hip',
            id=3,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        4:
        dict(
            name='left_knee',
            id=4,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        5:
        dict(
            name='left_ankle',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        6:
        dict(name='pelvis', id=6, color=[51, 153, 255], type='lower', swap=''),
        7:
        dict(name='thorax', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(
            name='upper_neck',
            id=8,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        9:
        dict(
            name='head_top', id=9, color=[51, 153, 255], type='upper',
            swap=''),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='right_elbow',
            id=11,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        12:
        dict(
            name='right_shoulder',
            id=12,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        13:
        dict(
            name='left_shoulder',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        14:
        dict(
            name='left_elbow',
            id=14,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        15:
        dict(
            name='left_wrist',
            id=15,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist')
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('right_knee', 'right_hip'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('right_hip', 'pelvis'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('pelvis', 'left_hip'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('left_hip', 'left_knee'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('left_knee', 'left_ankle'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('pelvis', 'thorax'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('thorax', 'upper_neck'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('upper_neck', 'head_top'), id=8, color=[51, 153, 255]),
        9:
        dict(link=('upper_neck', 'right_shoulder'), id=9, color=[255, 128, 0]),
        10:
        dict(
            link=('right_shoulder', 'right_elbow'), id=10, color=[255, 128,
                                                                  0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('upper_neck', 'left_shoulder'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('left_shoulder', 'left_elbow'), id=13, color=[0, 255, 0]),
        14:
        dict(link=('left_elbow', 'left_wrist'), id=14, color=[0, 255, 0])
    },
    joint_weights=[
        1.5, 1.2, 1., 1., 1.2, 1.5, 1., 1., 1., 1., 1.5, 1.2, 1., 1., 1.2, 1.5
    ],
    # Adapted from COCO dataset.
    sigmas=[
        0.089, 0.083, 0.107, 0.107, 0.083, 0.089, 0.026, 0.026, 0.026, 0.026,
        0.062, 0.072, 0.179, 0.179, 0.072, 0.062
    ])
