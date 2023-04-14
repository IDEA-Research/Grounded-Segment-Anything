dataset_info = dict(
    dataset_name='locust',
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
        dict(name='head', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='neck', id=1, color=[255, 255, 255], type='', swap=''),
        2:
        dict(name='thorax', id=2, color=[255, 255, 255], type='', swap=''),
        3:
        dict(name='abdomen1', id=3, color=[255, 255, 255], type='', swap=''),
        4:
        dict(name='abdomen2', id=4, color=[255, 255, 255], type='', swap=''),
        5:
        dict(
            name='anttipL',
            id=5,
            color=[255, 255, 255],
            type='',
            swap='anttipR'),
        6:
        dict(
            name='antbaseL',
            id=6,
            color=[255, 255, 255],
            type='',
            swap='antbaseR'),
        7:
        dict(name='eyeL', id=7, color=[255, 255, 255], type='', swap='eyeR'),
        8:
        dict(
            name='forelegL1',
            id=8,
            color=[255, 255, 255],
            type='',
            swap='forelegR1'),
        9:
        dict(
            name='forelegL2',
            id=9,
            color=[255, 255, 255],
            type='',
            swap='forelegR2'),
        10:
        dict(
            name='forelegL3',
            id=10,
            color=[255, 255, 255],
            type='',
            swap='forelegR3'),
        11:
        dict(
            name='forelegL4',
            id=11,
            color=[255, 255, 255],
            type='',
            swap='forelegR4'),
        12:
        dict(
            name='midlegL1',
            id=12,
            color=[255, 255, 255],
            type='',
            swap='midlegR1'),
        13:
        dict(
            name='midlegL2',
            id=13,
            color=[255, 255, 255],
            type='',
            swap='midlegR2'),
        14:
        dict(
            name='midlegL3',
            id=14,
            color=[255, 255, 255],
            type='',
            swap='midlegR3'),
        15:
        dict(
            name='midlegL4',
            id=15,
            color=[255, 255, 255],
            type='',
            swap='midlegR4'),
        16:
        dict(
            name='hindlegL1',
            id=16,
            color=[255, 255, 255],
            type='',
            swap='hindlegR1'),
        17:
        dict(
            name='hindlegL2',
            id=17,
            color=[255, 255, 255],
            type='',
            swap='hindlegR2'),
        18:
        dict(
            name='hindlegL3',
            id=18,
            color=[255, 255, 255],
            type='',
            swap='hindlegR3'),
        19:
        dict(
            name='hindlegL4',
            id=19,
            color=[255, 255, 255],
            type='',
            swap='hindlegR4'),
        20:
        dict(
            name='anttipR',
            id=20,
            color=[255, 255, 255],
            type='',
            swap='anttipL'),
        21:
        dict(
            name='antbaseR',
            id=21,
            color=[255, 255, 255],
            type='',
            swap='antbaseL'),
        22:
        dict(name='eyeR', id=22, color=[255, 255, 255], type='', swap='eyeL'),
        23:
        dict(
            name='forelegR1',
            id=23,
            color=[255, 255, 255],
            type='',
            swap='forelegL1'),
        24:
        dict(
            name='forelegR2',
            id=24,
            color=[255, 255, 255],
            type='',
            swap='forelegL2'),
        25:
        dict(
            name='forelegR3',
            id=25,
            color=[255, 255, 255],
            type='',
            swap='forelegL3'),
        26:
        dict(
            name='forelegR4',
            id=26,
            color=[255, 255, 255],
            type='',
            swap='forelegL4'),
        27:
        dict(
            name='midlegR1',
            id=27,
            color=[255, 255, 255],
            type='',
            swap='midlegL1'),
        28:
        dict(
            name='midlegR2',
            id=28,
            color=[255, 255, 255],
            type='',
            swap='midlegL2'),
        29:
        dict(
            name='midlegR3',
            id=29,
            color=[255, 255, 255],
            type='',
            swap='midlegL3'),
        30:
        dict(
            name='midlegR4',
            id=30,
            color=[255, 255, 255],
            type='',
            swap='midlegL4'),
        31:
        dict(
            name='hindlegR1',
            id=31,
            color=[255, 255, 255],
            type='',
            swap='hindlegL1'),
        32:
        dict(
            name='hindlegR2',
            id=32,
            color=[255, 255, 255],
            type='',
            swap='hindlegL2'),
        33:
        dict(
            name='hindlegR3',
            id=33,
            color=[255, 255, 255],
            type='',
            swap='hindlegL3'),
        34:
        dict(
            name='hindlegR4',
            id=34,
            color=[255, 255, 255],
            type='',
            swap='hindlegL4')
    },
    skeleton_info={
        0: dict(link=('neck', 'head'), id=0, color=[255, 255, 255]),
        1: dict(link=('thorax', 'neck'), id=1, color=[255, 255, 255]),
        2: dict(link=('abdomen1', 'thorax'), id=2, color=[255, 255, 255]),
        3: dict(link=('abdomen2', 'abdomen1'), id=3, color=[255, 255, 255]),
        4: dict(link=('antbaseL', 'anttipL'), id=4, color=[255, 255, 255]),
        5: dict(link=('eyeL', 'antbaseL'), id=5, color=[255, 255, 255]),
        6: dict(link=('forelegL2', 'forelegL1'), id=6, color=[255, 255, 255]),
        7: dict(link=('forelegL3', 'forelegL2'), id=7, color=[255, 255, 255]),
        8: dict(link=('forelegL4', 'forelegL3'), id=8, color=[255, 255, 255]),
        9: dict(link=('midlegL2', 'midlegL1'), id=9, color=[255, 255, 255]),
        10: dict(link=('midlegL3', 'midlegL2'), id=10, color=[255, 255, 255]),
        11: dict(link=('midlegL4', 'midlegL3'), id=11, color=[255, 255, 255]),
        12:
        dict(link=('hindlegL2', 'hindlegL1'), id=12, color=[255, 255, 255]),
        13:
        dict(link=('hindlegL3', 'hindlegL2'), id=13, color=[255, 255, 255]),
        14:
        dict(link=('hindlegL4', 'hindlegL3'), id=14, color=[255, 255, 255]),
        15: dict(link=('antbaseR', 'anttipR'), id=15, color=[255, 255, 255]),
        16: dict(link=('eyeR', 'antbaseR'), id=16, color=[255, 255, 255]),
        17:
        dict(link=('forelegR2', 'forelegR1'), id=17, color=[255, 255, 255]),
        18:
        dict(link=('forelegR3', 'forelegR2'), id=18, color=[255, 255, 255]),
        19:
        dict(link=('forelegR4', 'forelegR3'), id=19, color=[255, 255, 255]),
        20: dict(link=('midlegR2', 'midlegR1'), id=20, color=[255, 255, 255]),
        21: dict(link=('midlegR3', 'midlegR2'), id=21, color=[255, 255, 255]),
        22: dict(link=('midlegR4', 'midlegR3'), id=22, color=[255, 255, 255]),
        23:
        dict(link=('hindlegR2', 'hindlegR1'), id=23, color=[255, 255, 255]),
        24:
        dict(link=('hindlegR3', 'hindlegR2'), id=24, color=[255, 255, 255]),
        25:
        dict(link=('hindlegR4', 'hindlegR3'), id=25, color=[255, 255, 255])
    },
    joint_weights=[1.] * 35,
    sigmas=[])
