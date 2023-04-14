dataset_info = dict(
    dataset_name='panoptic_hand2d',
    paper_info=dict(
        author='Simon, Tomas and Joo, Hanbyul and '
        'Matthews, Iain and Sheikh, Yaser',
        title='Hand keypoint detection in single images using '
        'multiview bootstrapping',
        container='Proceedings of the IEEE conference on '
        'Computer Vision and Pattern Recognition',
        year='2017',
        homepage='http://domedb.perception.cs.cmu.edu/handdb.html',
    ),
    keypoint_info={
        0:
        dict(name='wrist', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='thumb1', id=1, color=[255, 128, 0], type='', swap=''),
        2:
        dict(name='thumb2', id=2, color=[255, 128, 0], type='', swap=''),
        3:
        dict(name='thumb3', id=3, color=[255, 128, 0], type='', swap=''),
        4:
        dict(name='thumb4', id=4, color=[255, 128, 0], type='', swap=''),
        5:
        dict(
            name='forefinger1', id=5, color=[255, 153, 255], type='', swap=''),
        6:
        dict(
            name='forefinger2', id=6, color=[255, 153, 255], type='', swap=''),
        7:
        dict(
            name='forefinger3', id=7, color=[255, 153, 255], type='', swap=''),
        8:
        dict(
            name='forefinger4', id=8, color=[255, 153, 255], type='', swap=''),
        9:
        dict(
            name='middle_finger1',
            id=9,
            color=[102, 178, 255],
            type='',
            swap=''),
        10:
        dict(
            name='middle_finger2',
            id=10,
            color=[102, 178, 255],
            type='',
            swap=''),
        11:
        dict(
            name='middle_finger3',
            id=11,
            color=[102, 178, 255],
            type='',
            swap=''),
        12:
        dict(
            name='middle_finger4',
            id=12,
            color=[102, 178, 255],
            type='',
            swap=''),
        13:
        dict(
            name='ring_finger1', id=13, color=[255, 51, 51], type='', swap=''),
        14:
        dict(
            name='ring_finger2', id=14, color=[255, 51, 51], type='', swap=''),
        15:
        dict(
            name='ring_finger3', id=15, color=[255, 51, 51], type='', swap=''),
        16:
        dict(
            name='ring_finger4', id=16, color=[255, 51, 51], type='', swap=''),
        17:
        dict(name='pinky_finger1', id=17, color=[0, 255, 0], type='', swap=''),
        18:
        dict(name='pinky_finger2', id=18, color=[0, 255, 0], type='', swap=''),
        19:
        dict(name='pinky_finger3', id=19, color=[0, 255, 0], type='', swap=''),
        20:
        dict(name='pinky_finger4', id=20, color=[0, 255, 0], type='', swap='')
    },
    skeleton_info={
        0:
        dict(link=('wrist', 'thumb1'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('thumb1', 'thumb2'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('thumb2', 'thumb3'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('thumb3', 'thumb4'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('wrist', 'forefinger1'), id=4, color=[255, 153, 255]),
        5:
        dict(link=('forefinger1', 'forefinger2'), id=5, color=[255, 153, 255]),
        6:
        dict(link=('forefinger2', 'forefinger3'), id=6, color=[255, 153, 255]),
        7:
        dict(link=('forefinger3', 'forefinger4'), id=7, color=[255, 153, 255]),
        8:
        dict(link=('wrist', 'middle_finger1'), id=8, color=[102, 178, 255]),
        9:
        dict(
            link=('middle_finger1', 'middle_finger2'),
            id=9,
            color=[102, 178, 255]),
        10:
        dict(
            link=('middle_finger2', 'middle_finger3'),
            id=10,
            color=[102, 178, 255]),
        11:
        dict(
            link=('middle_finger3', 'middle_finger4'),
            id=11,
            color=[102, 178, 255]),
        12:
        dict(link=('wrist', 'ring_finger1'), id=12, color=[255, 51, 51]),
        13:
        dict(
            link=('ring_finger1', 'ring_finger2'), id=13, color=[255, 51, 51]),
        14:
        dict(
            link=('ring_finger2', 'ring_finger3'), id=14, color=[255, 51, 51]),
        15:
        dict(
            link=('ring_finger3', 'ring_finger4'), id=15, color=[255, 51, 51]),
        16:
        dict(link=('wrist', 'pinky_finger1'), id=16, color=[0, 255, 0]),
        17:
        dict(
            link=('pinky_finger1', 'pinky_finger2'), id=17, color=[0, 255, 0]),
        18:
        dict(
            link=('pinky_finger2', 'pinky_finger3'), id=18, color=[0, 255, 0]),
        19:
        dict(
            link=('pinky_finger3', 'pinky_finger4'), id=19, color=[0, 255, 0])
    },
    joint_weights=[1.] * 21,
    sigmas=[])
