dataset_info = dict(
    dataset_name='nvgesture',
    paper_info=dict(
        author='Pavlo Molchanov and Xiaodong Yang and Shalini Gupta '
        'and Kihwan Kim and Stephen Tyree and Jan Kautz',
        title='Online Detection and Classification of Dynamic Hand Gestures '
        'with Recurrent 3D Convolutional Neural Networks',
        container='Proceedings of the IEEE Conference on '
        'Computer Vision and Pattern Recognition',
        year='2016',
        homepage='https://research.nvidia.com/publication/2016-06_online-'
        'detection-and-classification-dynamic-hand-gestures-recurrent-3d',
    ),
    category_info={
        0: 'five fingers move right',
        1: 'five fingers move left',
        2: 'five fingers move up',
        3: 'five fingers move down',
        4: 'two fingers move right',
        5: 'two fingers move left',
        6: 'two fingers move up',
        7: 'two fingers move down',
        8: 'click',
        9: 'beckoned',
        10: 'stretch hand',
        11: 'shake hand',
        12: 'one',
        13: 'two',
        14: 'three',
        15: 'lift up',
        16: 'press down',
        17: 'push',
        18: 'shrink',
        19: 'levorotation',
        20: 'dextrorotation',
        21: 'two fingers prod',
        22: 'grab',
        23: 'thumbs up',
        24: 'OK'
    },
    flip_pairs=[(0, 1), (4, 5), (19, 20)],
    fps=30)
