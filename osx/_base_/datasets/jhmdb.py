dataset_info = dict(
    dataset_name='jhmdb',
    paper_info=dict(
        author='H. Jhuang and J. Gall and S. Zuffi and '
        'C. Schmid and M. J. Black',
        title='Towards understanding action recognition',
        container='International Conf. on Computer Vision (ICCV)',
        year='2013',
        homepage='http://jhmdb.is.tue.mpg.de/dataset',
    ),
    keypoint_info={
        0:
        dict(name='neck', id=0, color=[255, 128, 0], type='upper', swap=''),
        1:
        dict(name='belly', id=1, color=[255, 128, 0], type='upper', swap=''),
        2:
        dict(name='head', id=2, color=[255, 128, 0], type='upper', swap=''),
        3:
        dict(
            name='right_shoulder',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='left_shoulder'),
        4:
        dict(
            name='left_shoulder',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        5:
        dict(
            name='right_hip',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='left_hip'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[51, 153, 255],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_elbow',
            id=7,
            color=[51, 153, 255],
            type='upper',
            swap='left_elbow'),
        8:
        dict(
            name='left_elbow',
            id=8,
            color=[51, 153, 255],
            type='upper',
            swap='right_elbow'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[51, 153, 255],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_knee',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='right_knee'),
        11:
        dict(
            name='right_wrist',
            id=11,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        12:
        dict(
            name='left_wrist',
            id=12,
            color=[255, 128, 0],
            type='upper',
            swap='right_wrist'),
        13:
        dict(
            name='right_ankle',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='left_ankle'),
        14:
        dict(
            name='left_ankle',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle')
    },
    skeleton_info={
        0: dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        1: dict(link=('right_knee', 'right_hip'), id=1, color=[255, 128, 0]),
        2: dict(link=('right_hip', 'belly'), id=2, color=[255, 128, 0]),
        3: dict(link=('belly', 'left_hip'), id=3, color=[0, 255, 0]),
        4: dict(link=('left_hip', 'left_knee'), id=4, color=[0, 255, 0]),
        5: dict(link=('left_knee', 'left_ankle'), id=5, color=[0, 255, 0]),
        6: dict(link=('belly', 'neck'), id=6, color=[51, 153, 255]),
        7: dict(link=('neck', 'head'), id=7, color=[51, 153, 255]),
        8: dict(link=('neck', 'right_shoulder'), id=8, color=[255, 128, 0]),
        9: dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('right_elbow', 'right_wrist'), id=10, color=[255, 128, 0]),
        11: dict(link=('neck', 'left_shoulder'), id=11, color=[0, 255, 0]),
        12:
        dict(link=('left_shoulder', 'left_elbow'), id=12, color=[0, 255, 0]),
        13: dict(link=('left_elbow', 'left_wrist'), id=13, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.5, 1.5
    ],
    # Adapted from COCO dataset.
    sigmas=[
        0.025, 0.107, 0.025, 0.079, 0.079, 0.107, 0.107, 0.072, 0.072, 0.087,
        0.087, 0.062, 0.062, 0.089, 0.089
    ])
