dataset_info = dict(
    dataset_name='crowdpose',
    paper_info=dict(
        author='Li, Jiefeng and Wang, Can and Zhu, Hao and '
        'Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu',
        title='CrowdPose: Efficient Crowded Scenes Pose Estimation '
        'and A New Benchmark',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2019',
        homepage='https://github.com/Jeff-sjtu/CrowdPose',
    ),
    keypoint_info={
        0:
        dict(
            name='left_shoulder',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='right_shoulder',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='left_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_wrist',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='right_wrist'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='left_knee',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='right_knee'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_ankle',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='right_ankle'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='left_ankle'),
        12:
        dict(
            name='top_head', id=12, color=[255, 128, 0], type='upper',
            swap=''),
        13:
        dict(name='neck', id=13, color=[0, 255, 0], type='upper', swap='')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('top_head', 'neck'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('right_shoulder', 'neck'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('left_shoulder', 'neck'), id=14, color=[51, 153, 255])
    },
    joint_weights=[
        0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5
    ],
    sigmas=[
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089, 0.079, 0.079
    ])
