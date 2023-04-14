dataset_info = dict(
    dataset_name='ap10k',
    paper_info=dict(
        author='Yu, Hang and Xu, Yufei and Zhang, Jing and '
        'Zhao, Wei and Guan, Ziyu and Tao, Dacheng',
        title='AP-10K: A Benchmark for Animal Pose Estimation in the Wild',
        container='35th Conference on Neural Information Processing Systems '
        '(NeurIPS 2021) Track on Datasets and Bench-marks.',
        year='2021',
        homepage='https://github.com/AlexTheBad/AP-10K',
    ),
    keypoint_info={
        0:
        dict(
            name='L_Eye', id=0, color=[0, 255, 0], type='upper', swap='R_Eye'),
        1:
        dict(
            name='R_Eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='L_Eye'),
        2:
        dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(name='Neck', id=3, color=[51, 153, 255], type='upper', swap=''),
        4:
        dict(
            name='Root of tail',
            id=4,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        5:
        dict(
            name='L_Shoulder',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='R_Shoulder'),
        6:
        dict(
            name='L_Elbow',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='R_Elbow'),
        7:
        dict(
            name='L_F_Paw',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Paw'),
        8:
        dict(
            name='R_Shoulder',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='L_Shoulder'),
        9:
        dict(
            name='R_Elbow',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='L_Elbow'),
        10:
        dict(
            name='R_F_Paw',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='L_F_Paw'),
        11:
        dict(
            name='L_Hip',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='R_Hip'),
        12:
        dict(
            name='L_Knee',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='R_Knee'),
        13:
        dict(
            name='L_B_Paw',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Paw'),
        14:
        dict(
            name='R_Hip', id=14, color=[0, 255, 0], type='lower',
            swap='L_Hip'),
        15:
        dict(
            name='R_Knee',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='L_Knee'),
        16:
        dict(
            name='R_B_Paw',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='L_B_Paw'),
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[0, 0, 255]),
        1: dict(link=('L_Eye', 'Nose'), id=1, color=[0, 0, 255]),
        2: dict(link=('R_Eye', 'Nose'), id=2, color=[0, 0, 255]),
        3: dict(link=('Nose', 'Neck'), id=3, color=[0, 255, 0]),
        4: dict(link=('Neck', 'Root of tail'), id=4, color=[0, 255, 0]),
        5: dict(link=('Neck', 'L_Shoulder'), id=5, color=[0, 255, 255]),
        6: dict(link=('L_Shoulder', 'L_Elbow'), id=6, color=[0, 255, 255]),
        7: dict(link=('L_Elbow', 'L_F_Paw'), id=6, color=[0, 255, 255]),
        8: dict(link=('Neck', 'R_Shoulder'), id=7, color=[6, 156, 250]),
        9: dict(link=('R_Shoulder', 'R_Elbow'), id=8, color=[6, 156, 250]),
        10: dict(link=('R_Elbow', 'R_F_Paw'), id=9, color=[6, 156, 250]),
        11: dict(link=('Root of tail', 'L_Hip'), id=10, color=[0, 255, 255]),
        12: dict(link=('L_Hip', 'L_Knee'), id=11, color=[0, 255, 255]),
        13: dict(link=('L_Knee', 'L_B_Paw'), id=12, color=[0, 255, 255]),
        14: dict(link=('Root of tail', 'R_Hip'), id=13, color=[6, 156, 250]),
        15: dict(link=('R_Hip', 'R_Knee'), id=14, color=[6, 156, 250]),
        16: dict(link=('R_Knee', 'R_B_Paw'), id=15, color=[6, 156, 250]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
    ])
