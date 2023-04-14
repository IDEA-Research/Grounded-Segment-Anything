dataset_info = dict(
    dataset_name='h36m',
    paper_info=dict(
        author='Ionescu, Catalin and Papava, Dragos and '
        'Olaru, Vlad and Sminchisescu, Cristian',
        title='Human3.6M: Large Scale Datasets and Predictive '
        'Methods for 3D Human Sensing in Natural Environments',
        container='IEEE Transactions on Pattern Analysis and '
        'Machine Intelligence',
        year='2014',
        homepage='http://vision.imar.ro/human3.6m/description.php',
    ),
    keypoint_info={
        0:
        dict(name='root', id=0, color=[51, 153, 255], type='lower', swap=''),
        1:
        dict(
            name='right_hip',
            id=1,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        2:
        dict(
            name='right_knee',
            id=2,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        3:
        dict(
            name='right_foot',
            id=3,
            color=[255, 128, 0],
            type='lower',
            swap='left_foot'),
        4:
        dict(
            name='left_hip',
            id=4,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        5:
        dict(
            name='left_knee',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        6:
        dict(
            name='left_foot',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap='right_foot'),
        7:
        dict(name='spine', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(name='thorax', id=8, color=[51, 153, 255], type='upper', swap=''),
        9:
        dict(
            name='neck_base',
            id=9,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        10:
        dict(name='head', id=10, color=[51, 153, 255], type='upper', swap=''),
        11:
        dict(
            name='left_shoulder',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        12:
        dict(
            name='left_elbow',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        13:
        dict(
            name='left_wrist',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        14:
        dict(
            name='right_shoulder',
            id=14,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        15:
        dict(
            name='right_elbow',
            id=15,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        16:
        dict(
            name='right_wrist',
            id=16,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist')
    },
    skeleton_info={
        0:
        dict(link=('root', 'left_hip'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_hip', 'left_knee'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_knee', 'left_foot'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('root', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_hip', 'right_knee'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('right_knee', 'right_foot'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('root', 'spine'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('spine', 'thorax'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('thorax', 'neck_base'), id=8, color=[51, 153, 255]),
        9:
        dict(link=('neck_base', 'head'), id=9, color=[51, 153, 255]),
        10:
        dict(link=('thorax', 'left_shoulder'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('left_shoulder', 'left_elbow'), id=11, color=[0, 255, 0]),
        12:
        dict(link=('left_elbow', 'left_wrist'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('thorax', 'right_shoulder'), id=13, color=[255, 128, 0]),
        14:
        dict(
            link=('right_shoulder', 'right_elbow'), id=14, color=[255, 128,
                                                                  0]),
        15:
        dict(link=('right_elbow', 'right_wrist'), id=15, color=[255, 128, 0])
    },
    joint_weights=[1.] * 17,
    sigmas=[],
    stats_info=dict(bbox_center=(528., 427.), bbox_scale=400.))
