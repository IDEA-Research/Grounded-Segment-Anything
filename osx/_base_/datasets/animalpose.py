dataset_info = dict(
    dataset_name='animalpose',
    paper_info=dict(
        author='Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and '
        'Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        container='The IEEE International Conference on '
        'Computer Vision (ICCV)',
        year='2019',
        homepage='https://sites.google.com/view/animal-pose/',
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
        dict(
            name='L_EarBase',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='R_EarBase'),
        3:
        dict(
            name='R_EarBase',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='L_EarBase'),
        4:
        dict(name='Nose', id=4, color=[51, 153, 255], type='upper', swap=''),
        5:
        dict(name='Throat', id=5, color=[51, 153, 255], type='upper', swap=''),
        6:
        dict(
            name='TailBase', id=6, color=[51, 153, 255], type='lower',
            swap=''),
        7:
        dict(
            name='Withers', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(
            name='L_F_Elbow',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Elbow'),
        9:
        dict(
            name='R_F_Elbow',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_Elbow'),
        10:
        dict(
            name='L_B_Elbow',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Elbow'),
        11:
        dict(
            name='R_B_Elbow',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_Elbow'),
        12:
        dict(
            name='L_F_Knee',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Knee'),
        13:
        dict(
            name='R_F_Knee',
            id=13,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_Knee'),
        14:
        dict(
            name='L_B_Knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Knee'),
        15:
        dict(
            name='R_B_Knee',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_Knee'),
        16:
        dict(
            name='L_F_Paw',
            id=16,
            color=[0, 255, 0],
            type='upper',
            swap='R_F_Paw'),
        17:
        dict(
            name='R_F_Paw',
            id=17,
            color=[255, 128, 0],
            type='upper',
            swap='L_F_Paw'),
        18:
        dict(
            name='L_B_Paw',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap='R_B_Paw'),
        19:
        dict(
            name='R_B_Paw',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='L_B_Paw')
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[51, 153, 255]),
        1: dict(link=('L_Eye', 'L_EarBase'), id=1, color=[0, 255, 0]),
        2: dict(link=('R_Eye', 'R_EarBase'), id=2, color=[255, 128, 0]),
        3: dict(link=('L_Eye', 'Nose'), id=3, color=[0, 255, 0]),
        4: dict(link=('R_Eye', 'Nose'), id=4, color=[255, 128, 0]),
        5: dict(link=('Nose', 'Throat'), id=5, color=[51, 153, 255]),
        6: dict(link=('Throat', 'Withers'), id=6, color=[51, 153, 255]),
        7: dict(link=('TailBase', 'Withers'), id=7, color=[51, 153, 255]),
        8: dict(link=('Throat', 'L_F_Elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('L_F_Elbow', 'L_F_Knee'), id=9, color=[0, 255, 0]),
        10: dict(link=('L_F_Knee', 'L_F_Paw'), id=10, color=[0, 255, 0]),
        11: dict(link=('Throat', 'R_F_Elbow'), id=11, color=[255, 128, 0]),
        12: dict(link=('R_F_Elbow', 'R_F_Knee'), id=12, color=[255, 128, 0]),
        13: dict(link=('R_F_Knee', 'R_F_Paw'), id=13, color=[255, 128, 0]),
        14: dict(link=('TailBase', 'L_B_Elbow'), id=14, color=[0, 255, 0]),
        15: dict(link=('L_B_Elbow', 'L_B_Knee'), id=15, color=[0, 255, 0]),
        16: dict(link=('L_B_Knee', 'L_B_Paw'), id=16, color=[0, 255, 0]),
        17: dict(link=('TailBase', 'R_B_Elbow'), id=17, color=[255, 128, 0]),
        18: dict(link=('R_B_Elbow', 'R_B_Knee'), id=18, color=[255, 128, 0]),
        19: dict(link=('R_B_Knee', 'R_B_Paw'), id=19, color=[255, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2,
        1.5, 1.5, 1.5, 1.5
    ],

    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.10, 0.107, 0.107,
        0.107, 0.107, 0.087, 0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089
    ])
