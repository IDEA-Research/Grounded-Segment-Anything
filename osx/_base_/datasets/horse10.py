dataset_info = dict(
    dataset_name='horse10',
    paper_info=dict(
        author='Mathis, Alexander and Biasi, Thomas and '
        'Schneider, Steffen and '
        'Yuksekgonul, Mert and Rogers, Byron and '
        'Bethge, Matthias and '
        'Mathis, Mackenzie W',
        title='Pretraining boosts out-of-domain robustness '
        'for pose estimation',
        container='Proceedings of the IEEE/CVF Winter Conference on '
        'Applications of Computer Vision',
        year='2021',
        homepage='http://www.mackenziemathislab.org/horse10',
    ),
    keypoint_info={
        0:
        dict(name='Nose', id=0, color=[255, 153, 255], type='upper', swap=''),
        1:
        dict(name='Eye', id=1, color=[255, 153, 255], type='upper', swap=''),
        2:
        dict(
            name='Nearknee',
            id=2,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='Nearfrontfetlock',
            id=3,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='Nearfrontfoot',
            id=4,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='Offknee', id=5, color=[255, 102, 255], type='upper',
            swap=''),
        6:
        dict(
            name='Offfrontfetlock',
            id=6,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        7:
        dict(
            name='Offfrontfoot',
            id=7,
            color=[255, 102, 255],
            type='upper',
            swap=''),
        8:
        dict(
            name='Shoulder',
            id=8,
            color=[255, 153, 255],
            type='upper',
            swap=''),
        9:
        dict(
            name='Midshoulder',
            id=9,
            color=[255, 153, 255],
            type='upper',
            swap=''),
        10:
        dict(
            name='Elbow', id=10, color=[255, 153, 255], type='upper', swap=''),
        11:
        dict(
            name='Girth', id=11, color=[255, 153, 255], type='upper', swap=''),
        12:
        dict(
            name='Wither', id=12, color=[255, 153, 255], type='upper',
            swap=''),
        13:
        dict(
            name='Nearhindhock',
            id=13,
            color=[255, 51, 255],
            type='lower',
            swap=''),
        14:
        dict(
            name='Nearhindfetlock',
            id=14,
            color=[255, 51, 255],
            type='lower',
            swap=''),
        15:
        dict(
            name='Nearhindfoot',
            id=15,
            color=[255, 51, 255],
            type='lower',
            swap=''),
        16:
        dict(name='Hip', id=16, color=[255, 153, 255], type='lower', swap=''),
        17:
        dict(
            name='Stifle', id=17, color=[255, 153, 255], type='lower',
            swap=''),
        18:
        dict(
            name='Offhindhock',
            id=18,
            color=[255, 51, 255],
            type='lower',
            swap=''),
        19:
        dict(
            name='Offhindfetlock',
            id=19,
            color=[255, 51, 255],
            type='lower',
            swap=''),
        20:
        dict(
            name='Offhindfoot',
            id=20,
            color=[255, 51, 255],
            type='lower',
            swap=''),
        21:
        dict(
            name='Ischium',
            id=21,
            color=[255, 153, 255],
            type='lower',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('Nose', 'Eye'), id=0, color=[255, 153, 255]),
        1:
        dict(link=('Eye', 'Wither'), id=1, color=[255, 153, 255]),
        2:
        dict(link=('Wither', 'Hip'), id=2, color=[255, 153, 255]),
        3:
        dict(link=('Hip', 'Ischium'), id=3, color=[255, 153, 255]),
        4:
        dict(link=('Ischium', 'Stifle'), id=4, color=[255, 153, 255]),
        5:
        dict(link=('Stifle', 'Girth'), id=5, color=[255, 153, 255]),
        6:
        dict(link=('Girth', 'Elbow'), id=6, color=[255, 153, 255]),
        7:
        dict(link=('Elbow', 'Shoulder'), id=7, color=[255, 153, 255]),
        8:
        dict(link=('Shoulder', 'Midshoulder'), id=8, color=[255, 153, 255]),
        9:
        dict(link=('Midshoulder', 'Wither'), id=9, color=[255, 153, 255]),
        10:
        dict(
            link=('Nearknee', 'Nearfrontfetlock'),
            id=10,
            color=[255, 102, 255]),
        11:
        dict(
            link=('Nearfrontfetlock', 'Nearfrontfoot'),
            id=11,
            color=[255, 102, 255]),
        12:
        dict(
            link=('Offknee', 'Offfrontfetlock'), id=12, color=[255, 102, 255]),
        13:
        dict(
            link=('Offfrontfetlock', 'Offfrontfoot'),
            id=13,
            color=[255, 102, 255]),
        14:
        dict(
            link=('Nearhindhock', 'Nearhindfetlock'),
            id=14,
            color=[255, 51, 255]),
        15:
        dict(
            link=('Nearhindfetlock', 'Nearhindfoot'),
            id=15,
            color=[255, 51, 255]),
        16:
        dict(
            link=('Offhindhock', 'Offhindfetlock'),
            id=16,
            color=[255, 51, 255]),
        17:
        dict(
            link=('Offhindfetlock', 'Offhindfoot'),
            id=17,
            color=[255, 51, 255])
    },
    joint_weights=[1.] * 22,
    sigmas=[])
