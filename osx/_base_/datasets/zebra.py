dataset_info = dict(
    dataset_name='zebra',
    paper_info=dict(
        author='Graving, Jacob M and Chae, Daniel and Naik, Hemal and '
        'Li, Liang and Koger, Benjamin and Costelloe, Blair R and '
        'Couzin, Iain D',
        title='DeepPoseKit, a software toolkit for fast and robust '
        'animal pose estimation using deep learning',
        container='Elife',
        year='2019',
        homepage='https://github.com/jgraving/DeepPoseKit-Data',
    ),
    keypoint_info={
        0:
        dict(name='snout', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='head', id=1, color=[255, 255, 255], type='', swap=''),
        2:
        dict(name='neck', id=2, color=[255, 255, 255], type='', swap=''),
        3:
        dict(
            name='forelegL1',
            id=3,
            color=[255, 255, 255],
            type='',
            swap='forelegR1'),
        4:
        dict(
            name='forelegR1',
            id=4,
            color=[255, 255, 255],
            type='',
            swap='forelegL1'),
        5:
        dict(
            name='hindlegL1',
            id=5,
            color=[255, 255, 255],
            type='',
            swap='hindlegR1'),
        6:
        dict(
            name='hindlegR1',
            id=6,
            color=[255, 255, 255],
            type='',
            swap='hindlegL1'),
        7:
        dict(name='tailbase', id=7, color=[255, 255, 255], type='', swap=''),
        8:
        dict(name='tailtip', id=8, color=[255, 255, 255], type='', swap='')
    },
    skeleton_info={
        0: dict(link=('head', 'snout'), id=0, color=[255, 255, 255]),
        1: dict(link=('neck', 'head'), id=1, color=[255, 255, 255]),
        2: dict(link=('forelegL1', 'neck'), id=2, color=[255, 255, 255]),
        3: dict(link=('forelegR1', 'neck'), id=3, color=[255, 255, 255]),
        4: dict(link=('hindlegL1', 'tailbase'), id=4, color=[255, 255, 255]),
        5: dict(link=('hindlegR1', 'tailbase'), id=5, color=[255, 255, 255]),
        6: dict(link=('tailbase', 'neck'), id=6, color=[255, 255, 255]),
        7: dict(link=('tailtip', 'tailbase'), id=7, color=[255, 255, 255])
    },
    joint_weights=[1.] * 9,
    sigmas=[])
