dataset_info = dict(
    dataset_name='fly',
    paper_info=dict(
        author='Pereira, Talmo D and Aldarondo, Diego E and '
        'Willmore, Lindsay and Kislin, Mikhail and '
        'Wang, Samuel S-H and Murthy, Mala and Shaevitz, Joshua W',
        title='Fast animal pose estimation using deep neural networks',
        container='Nature methods',
        year='2019',
        homepage='https://github.com/jgraving/DeepPoseKit-Data',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='eyeL', id=1, color=[255, 255, 255], type='', swap='eyeR'),
        2:
        dict(name='eyeR', id=2, color=[255, 255, 255], type='', swap='eyeL'),
        3:
        dict(name='neck', id=3, color=[255, 255, 255], type='', swap=''),
        4:
        dict(name='thorax', id=4, color=[255, 255, 255], type='', swap=''),
        5:
        dict(name='abdomen', id=5, color=[255, 255, 255], type='', swap=''),
        6:
        dict(
            name='forelegR1',
            id=6,
            color=[255, 255, 255],
            type='',
            swap='forelegL1'),
        7:
        dict(
            name='forelegR2',
            id=7,
            color=[255, 255, 255],
            type='',
            swap='forelegL2'),
        8:
        dict(
            name='forelegR3',
            id=8,
            color=[255, 255, 255],
            type='',
            swap='forelegL3'),
        9:
        dict(
            name='forelegR4',
            id=9,
            color=[255, 255, 255],
            type='',
            swap='forelegL4'),
        10:
        dict(
            name='midlegR1',
            id=10,
            color=[255, 255, 255],
            type='',
            swap='midlegL1'),
        11:
        dict(
            name='midlegR2',
            id=11,
            color=[255, 255, 255],
            type='',
            swap='midlegL2'),
        12:
        dict(
            name='midlegR3',
            id=12,
            color=[255, 255, 255],
            type='',
            swap='midlegL3'),
        13:
        dict(
            name='midlegR4',
            id=13,
            color=[255, 255, 255],
            type='',
            swap='midlegL4'),
        14:
        dict(
            name='hindlegR1',
            id=14,
            color=[255, 255, 255],
            type='',
            swap='hindlegL1'),
        15:
        dict(
            name='hindlegR2',
            id=15,
            color=[255, 255, 255],
            type='',
            swap='hindlegL2'),
        16:
        dict(
            name='hindlegR3',
            id=16,
            color=[255, 255, 255],
            type='',
            swap='hindlegL3'),
        17:
        dict(
            name='hindlegR4',
            id=17,
            color=[255, 255, 255],
            type='',
            swap='hindlegL4'),
        18:
        dict(
            name='forelegL1',
            id=18,
            color=[255, 255, 255],
            type='',
            swap='forelegR1'),
        19:
        dict(
            name='forelegL2',
            id=19,
            color=[255, 255, 255],
            type='',
            swap='forelegR2'),
        20:
        dict(
            name='forelegL3',
            id=20,
            color=[255, 255, 255],
            type='',
            swap='forelegR3'),
        21:
        dict(
            name='forelegL4',
            id=21,
            color=[255, 255, 255],
            type='',
            swap='forelegR4'),
        22:
        dict(
            name='midlegL1',
            id=22,
            color=[255, 255, 255],
            type='',
            swap='midlegR1'),
        23:
        dict(
            name='midlegL2',
            id=23,
            color=[255, 255, 255],
            type='',
            swap='midlegR2'),
        24:
        dict(
            name='midlegL3',
            id=24,
            color=[255, 255, 255],
            type='',
            swap='midlegR3'),
        25:
        dict(
            name='midlegL4',
            id=25,
            color=[255, 255, 255],
            type='',
            swap='midlegR4'),
        26:
        dict(
            name='hindlegL1',
            id=26,
            color=[255, 255, 255],
            type='',
            swap='hindlegR1'),
        27:
        dict(
            name='hindlegL2',
            id=27,
            color=[255, 255, 255],
            type='',
            swap='hindlegR2'),
        28:
        dict(
            name='hindlegL3',
            id=28,
            color=[255, 255, 255],
            type='',
            swap='hindlegR3'),
        29:
        dict(
            name='hindlegL4',
            id=29,
            color=[255, 255, 255],
            type='',
            swap='hindlegR4'),
        30:
        dict(
            name='wingL', id=30, color=[255, 255, 255], type='', swap='wingR'),
        31:
        dict(
            name='wingR', id=31, color=[255, 255, 255], type='', swap='wingL'),
    },
    skeleton_info={
        0: dict(link=('eyeL', 'head'), id=0, color=[255, 255, 255]),
        1: dict(link=('eyeR', 'head'), id=1, color=[255, 255, 255]),
        2: dict(link=('neck', 'head'), id=2, color=[255, 255, 255]),
        3: dict(link=('thorax', 'neck'), id=3, color=[255, 255, 255]),
        4: dict(link=('abdomen', 'thorax'), id=4, color=[255, 255, 255]),
        5: dict(link=('forelegR2', 'forelegR1'), id=5, color=[255, 255, 255]),
        6: dict(link=('forelegR3', 'forelegR2'), id=6, color=[255, 255, 255]),
        7: dict(link=('forelegR4', 'forelegR3'), id=7, color=[255, 255, 255]),
        8: dict(link=('midlegR2', 'midlegR1'), id=8, color=[255, 255, 255]),
        9: dict(link=('midlegR3', 'midlegR2'), id=9, color=[255, 255, 255]),
        10: dict(link=('midlegR4', 'midlegR3'), id=10, color=[255, 255, 255]),
        11:
        dict(link=('hindlegR2', 'hindlegR1'), id=11, color=[255, 255, 255]),
        12:
        dict(link=('hindlegR3', 'hindlegR2'), id=12, color=[255, 255, 255]),
        13:
        dict(link=('hindlegR4', 'hindlegR3'), id=13, color=[255, 255, 255]),
        14:
        dict(link=('forelegL2', 'forelegL1'), id=14, color=[255, 255, 255]),
        15:
        dict(link=('forelegL3', 'forelegL2'), id=15, color=[255, 255, 255]),
        16:
        dict(link=('forelegL4', 'forelegL3'), id=16, color=[255, 255, 255]),
        17: dict(link=('midlegL2', 'midlegL1'), id=17, color=[255, 255, 255]),
        18: dict(link=('midlegL3', 'midlegL2'), id=18, color=[255, 255, 255]),
        19: dict(link=('midlegL4', 'midlegL3'), id=19, color=[255, 255, 255]),
        20:
        dict(link=('hindlegL2', 'hindlegL1'), id=20, color=[255, 255, 255]),
        21:
        dict(link=('hindlegL3', 'hindlegL2'), id=21, color=[255, 255, 255]),
        22:
        dict(link=('hindlegL4', 'hindlegL3'), id=22, color=[255, 255, 255]),
        23: dict(link=('wingL', 'neck'), id=23, color=[255, 255, 255]),
        24: dict(link=('wingR', 'neck'), id=24, color=[255, 255, 255])
    },
    joint_weights=[1.] * 32,
    sigmas=[])
