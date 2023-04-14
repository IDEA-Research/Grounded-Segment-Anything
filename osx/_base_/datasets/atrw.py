dataset_info = dict(
    dataset_name='atrw',
    paper_info=dict(
        author='Li, Shuyuan and Li, Jianguo and Tang, Hanlin '
        'and Qian, Rui and Lin, Weiyao',
        title='ATRW: A Benchmark for Amur Tiger '
        'Re-identification in the Wild',
        container='Proceedings of the 28th ACM '
        'International Conference on Multimedia',
        year='2020',
        homepage='https://cvwc2019.github.io/challenge.html',
    ),
    keypoint_info={
        0:
        dict(
            name='left_ear',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        1:
        dict(
            name='right_ear',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        2:
        dict(name='nose', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(
            name='right_shoulder',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        4:
        dict(
            name='right_front_paw',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_front_paw'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='left_front_paw',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_front_paw'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='right_knee',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        9:
        dict(
            name='right_back_paw',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_back_paw'),
        10:
        dict(
            name='left_hip',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        11:
        dict(
            name='left_knee',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        12:
        dict(
            name='left_back_paw',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='right_back_paw'),
        13:
        dict(name='tail', id=13, color=[51, 153, 255], type='lower', swap=''),
        14:
        dict(
            name='center', id=14, color=[51, 153, 255], type='lower', swap=''),
    },
    skeleton_info={
        0:
        dict(link=('left_ear', 'nose'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('right_ear', 'nose'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('nose', 'center'), id=2, color=[51, 153, 255]),
        3:
        dict(
            link=('left_shoulder', 'left_front_paw'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('left_shoulder', 'center'), id=4, color=[0, 255, 0]),
        5:
        dict(
            link=('right_shoulder', 'right_front_paw'),
            id=5,
            color=[255, 128, 0]),
        6:
        dict(link=('right_shoulder', 'center'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('tail', 'center'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('right_back_paw', 'right_knee'), id=8, color=[255, 128, 0]),
        9:
        dict(link=('right_knee', 'right_hip'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('right_hip', 'tail'), id=10, color=[255, 128, 0]),
        11:
        dict(link=('left_back_paw', 'left_knee'), id=11, color=[0, 255, 0]),
        12:
        dict(link=('left_knee', 'left_hip'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('left_hip', 'tail'), id=13, color=[0, 255, 0]),
    },
    joint_weights=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    sigmas=[
        0.0277, 0.0823, 0.0831, 0.0202, 0.0716, 0.0263, 0.0646, 0.0302, 0.0440,
        0.0316, 0.0333, 0.0547, 0.0263, 0.0683, 0.0539
    ])
