dataset_info = dict(
    dataset_name='campus',
    paper_info=dict(
        author='Belagiannis, Vasileios and Amin, Sikandar and Andriluka, '
        'Mykhaylo and Schiele, Bernt and Navab, Nassir and Ilic, Slobodan',
        title='3D Pictorial Structures for Multiple Human Pose Estimation',
        container='IEEE Computer Society Conference on Computer Vision and '
        'Pattern Recognition (CVPR)',
        year='2014',
        homepage='http://campar.in.tum.de/Chair/MultiHumanPose',
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
        dict(
            name='right_wrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        7:
        dict(
            name='right_elbow',
            id=7,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        8:
        dict(
            name='right_shoulder',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        9:
        dict(
            name='left_shoulder',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        10:
        dict(
            name='left_elbow',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        11:
        dict(
            name='left_wrist',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        12:
        dict(
            name='bottom_head',
            id=12,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        13:
        dict(
            name='top_head',
            id=13,
            color=[51, 153, 255],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('right_knee', 'right_hip'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('left_hip', 'left_knee'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('left_knee', 'left_ankle'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('right_hip', 'left_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('right_wrist', 'right_elbow'), id=5, color=[255, 128, 0]),
        6:
        dict(
            link=('right_elbow', 'right_shoulder'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('left_shoulder', 'left_elbow'), id=7, color=[0, 255, 0]),
        8:
        dict(link=('left_elbow', 'left_wrist'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('right_hip', 'right_shoulder'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_hip', 'left_shoulder'), id=10, color=[0, 255, 0]),
        11:
        dict(
            link=('right_shoulder', 'bottom_head'), id=11, color=[255, 128,
                                                                  0]),
        12:
        dict(link=('left_shoulder', 'bottom_head'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('bottom_head', 'top_head'), id=13, color=[51, 153, 255]),
    },
    joint_weights=[
        1.5, 1.2, 1.0, 1.0, 1.2, 1.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.5, 1.0, 1.0
    ],
    sigmas=[
        0.089, 0.087, 0.107, 0.107, 0.087, 0.089, 0.062, 0.072, 0.079, 0.079,
        0.072, 0.062, 0.026, 0.026
    ])
