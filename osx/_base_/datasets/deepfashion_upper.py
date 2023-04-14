dataset_info = dict(
    dataset_name='deepfashion_upper',
    paper_info=dict(
        author='Liu, Ziwei and Luo, Ping and Qiu, Shi '
        'and Wang, Xiaogang and Tang, Xiaoou',
        title='DeepFashion: Powering Robust Clothes Recognition '
        'and Retrieval with Rich Annotations',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2016',
        homepage='http://mmlab.ie.cuhk.edu.hk/projects/'
        'DeepFashion/LandmarkDetection.html',
    ),
    keypoint_info={
        0:
        dict(
            name='left collar',
            id=0,
            color=[255, 255, 255],
            type='',
            swap='right collar'),
        1:
        dict(
            name='right collar',
            id=1,
            color=[255, 255, 255],
            type='',
            swap='left collar'),
        2:
        dict(
            name='left sleeve',
            id=2,
            color=[255, 255, 255],
            type='',
            swap='right sleeve'),
        3:
        dict(
            name='right sleeve',
            id=3,
            color=[255, 255, 255],
            type='',
            swap='left sleeve'),
        4:
        dict(
            name='left hem',
            id=4,
            color=[255, 255, 255],
            type='',
            swap='right hem'),
        5:
        dict(
            name='right hem',
            id=5,
            color=[255, 255, 255],
            type='',
            swap='left hem'),
    },
    skeleton_info={},
    joint_weights=[1.] * 6,
    sigmas=[])
