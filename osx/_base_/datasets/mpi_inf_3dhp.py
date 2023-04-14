dataset_info = dict(
    dataset_name='mpi_inf_3dhp',
    paper_info=dict(
        author='ehta, Dushyant and Rhodin, Helge and Casas, Dan and '
        'Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and '
        'Theobalt, Christian',
        title='Monocular 3D Human Pose Estimation In The Wild Using Improved '
        'CNN Supervision',
        container='2017 international conference on 3D vision (3DV)',
        year='2017',
        homepage='http://gvv.mpi-inf.mpg.de/3dhp-dataset',
    ),
    keypoint_info={
        0:
        dict(
            name='head_top', id=0, color=[51, 153, 255], type='upper',
            swap=''),
        1:
        dict(name='neck', id=1, color=[51, 153, 255], type='upper', swap=''),
        2:
        dict(
            name='right_shoulder',
            id=2,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='right_wrist',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='left_elbow',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        7:
        dict(
            name='left_wrist',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        8:
        dict(
            name='right_hip',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='right_ankle',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='left_knee',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        13:
        dict(
            name='left_ankle',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        14:
        dict(name='root', id=14, color=[51, 153, 255], type='lower', swap=''),
        15:
        dict(name='spine', id=15, color=[51, 153, 255], type='upper', swap=''),
        16:
        dict(name='head', id=16, color=[51, 153, 255], type='upper', swap='')
    },
    skeleton_info={
        0: dict(link=('neck', 'right_shoulder'), id=0, color=[255, 128, 0]),
        1: dict(
            link=('right_shoulder', 'right_elbow'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('right_elbow', 'right_wrist'), id=2, color=[255, 128, 0]),
        3: dict(link=('neck', 'left_shoulder'), id=3, color=[0, 255, 0]),
        4: dict(link=('left_shoulder', 'left_elbow'), id=4, color=[0, 255, 0]),
        5: dict(link=('left_elbow', 'left_wrist'), id=5, color=[0, 255, 0]),
        6: dict(link=('root', 'right_hip'), id=6, color=[255, 128, 0]),
        7: dict(link=('right_hip', 'right_knee'), id=7, color=[255, 128, 0]),
        8: dict(link=('right_knee', 'right_ankle'), id=8, color=[255, 128, 0]),
        9: dict(link=('root', 'left_hip'), id=9, color=[0, 255, 0]),
        10: dict(link=('left_hip', 'left_knee'), id=10, color=[0, 255, 0]),
        11: dict(link=('left_knee', 'left_ankle'), id=11, color=[0, 255, 0]),
        12: dict(link=('head_top', 'head'), id=12, color=[51, 153, 255]),
        13: dict(link=('head', 'neck'), id=13, color=[51, 153, 255]),
        14: dict(link=('neck', 'spine'), id=14, color=[51, 153, 255]),
        15: dict(link=('spine', 'root'), id=15, color=[51, 153, 255])
    },
    joint_weights=[1.] * 17,
    sigmas=[])
