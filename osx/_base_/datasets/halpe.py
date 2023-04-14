dataset_info = dict(
    dataset_name='halpe',
    paper_info=dict(
        author='Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie'
        ' and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu'
        ' and Ma, Ze and Chen, Mingyang and Lu, Cewu',
        title='PaStaNet: Toward Human Activity Knowledge Engine',
        container='CVPR',
        year='2020',
        homepage='https://github.com/Fang-Haoshu/Halpe-FullBody/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(name='head', id=17, color=[255, 128, 0], type='upper', swap=''),
        18:
        dict(name='neck', id=18, color=[255, 128, 0], type='upper', swap=''),
        19:
        dict(name='hip', id=19, color=[255, 128, 0], type='lower', swap=''),
        20:
        dict(
            name='left_big_toe',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='right_big_toe'),
        21:
        dict(
            name='right_big_toe',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        22:
        dict(
            name='left_small_toe',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='right_small_toe'),
        23:
        dict(
            name='right_small_toe',
            id=23,
            color=[255, 128, 0],
            type='lower',
            swap='left_small_toe'),
        24:
        dict(
            name='left_heel',
            id=24,
            color=[255, 128, 0],
            type='lower',
            swap='right_heel'),
        25:
        dict(
            name='right_heel',
            id=25,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel'),
        26:
        dict(
            name='face-0',
            id=26,
            color=[255, 255, 255],
            type='',
            swap='face-16'),
        27:
        dict(
            name='face-1',
            id=27,
            color=[255, 255, 255],
            type='',
            swap='face-15'),
        28:
        dict(
            name='face-2',
            id=28,
            color=[255, 255, 255],
            type='',
            swap='face-14'),
        29:
        dict(
            name='face-3',
            id=29,
            color=[255, 255, 255],
            type='',
            swap='face-13'),
        30:
        dict(
            name='face-4',
            id=30,
            color=[255, 255, 255],
            type='',
            swap='face-12'),
        31:
        dict(
            name='face-5',
            id=31,
            color=[255, 255, 255],
            type='',
            swap='face-11'),
        32:
        dict(
            name='face-6',
            id=32,
            color=[255, 255, 255],
            type='',
            swap='face-10'),
        33:
        dict(
            name='face-7',
            id=33,
            color=[255, 255, 255],
            type='',
            swap='face-9'),
        34:
        dict(name='face-8', id=34, color=[255, 255, 255], type='', swap=''),
        35:
        dict(
            name='face-9',
            id=35,
            color=[255, 255, 255],
            type='',
            swap='face-7'),
        36:
        dict(
            name='face-10',
            id=36,
            color=[255, 255, 255],
            type='',
            swap='face-6'),
        37:
        dict(
            name='face-11',
            id=37,
            color=[255, 255, 255],
            type='',
            swap='face-5'),
        38:
        dict(
            name='face-12',
            id=38,
            color=[255, 255, 255],
            type='',
            swap='face-4'),
        39:
        dict(
            name='face-13',
            id=39,
            color=[255, 255, 255],
            type='',
            swap='face-3'),
        40:
        dict(
            name='face-14',
            id=40,
            color=[255, 255, 255],
            type='',
            swap='face-2'),
        41:
        dict(
            name='face-15',
            id=41,
            color=[255, 255, 255],
            type='',
            swap='face-1'),
        42:
        dict(
            name='face-16',
            id=42,
            color=[255, 255, 255],
            type='',
            swap='face-0'),
        43:
        dict(
            name='face-17',
            id=43,
            color=[255, 255, 255],
            type='',
            swap='face-26'),
        44:
        dict(
            name='face-18',
            id=44,
            color=[255, 255, 255],
            type='',
            swap='face-25'),
        45:
        dict(
            name='face-19',
            id=45,
            color=[255, 255, 255],
            type='',
            swap='face-24'),
        46:
        dict(
            name='face-20',
            id=46,
            color=[255, 255, 255],
            type='',
            swap='face-23'),
        47:
        dict(
            name='face-21',
            id=47,
            color=[255, 255, 255],
            type='',
            swap='face-22'),
        48:
        dict(
            name='face-22',
            id=48,
            color=[255, 255, 255],
            type='',
            swap='face-21'),
        49:
        dict(
            name='face-23',
            id=49,
            color=[255, 255, 255],
            type='',
            swap='face-20'),
        50:
        dict(
            name='face-24',
            id=50,
            color=[255, 255, 255],
            type='',
            swap='face-19'),
        51:
        dict(
            name='face-25',
            id=51,
            color=[255, 255, 255],
            type='',
            swap='face-18'),
        52:
        dict(
            name='face-26',
            id=52,
            color=[255, 255, 255],
            type='',
            swap='face-17'),
        53:
        dict(name='face-27', id=53, color=[255, 255, 255], type='', swap=''),
        54:
        dict(name='face-28', id=54, color=[255, 255, 255], type='', swap=''),
        55:
        dict(name='face-29', id=55, color=[255, 255, 255], type='', swap=''),
        56:
        dict(name='face-30', id=56, color=[255, 255, 255], type='', swap=''),
        57:
        dict(
            name='face-31',
            id=57,
            color=[255, 255, 255],
            type='',
            swap='face-35'),
        58:
        dict(
            name='face-32',
            id=58,
            color=[255, 255, 255],
            type='',
            swap='face-34'),
        59:
        dict(name='face-33', id=59, color=[255, 255, 255], type='', swap=''),
        60:
        dict(
            name='face-34',
            id=60,
            color=[255, 255, 255],
            type='',
            swap='face-32'),
        61:
        dict(
            name='face-35',
            id=61,
            color=[255, 255, 255],
            type='',
            swap='face-31'),
        62:
        dict(
            name='face-36',
            id=62,
            color=[255, 255, 255],
            type='',
            swap='face-45'),
        63:
        dict(
            name='face-37',
            id=63,
            color=[255, 255, 255],
            type='',
            swap='face-44'),
        64:
        dict(
            name='face-38',
            id=64,
            color=[255, 255, 255],
            type='',
            swap='face-43'),
        65:
        dict(
            name='face-39',
            id=65,
            color=[255, 255, 255],
            type='',
            swap='face-42'),
        66:
        dict(
            name='face-40',
            id=66,
            color=[255, 255, 255],
            type='',
            swap='face-47'),
        67:
        dict(
            name='face-41',
            id=67,
            color=[255, 255, 255],
            type='',
            swap='face-46'),
        68:
        dict(
            name='face-42',
            id=68,
            color=[255, 255, 255],
            type='',
            swap='face-39'),
        69:
        dict(
            name='face-43',
            id=69,
            color=[255, 255, 255],
            type='',
            swap='face-38'),
        70:
        dict(
            name='face-44',
            id=70,
            color=[255, 255, 255],
            type='',
            swap='face-37'),
        71:
        dict(
            name='face-45',
            id=71,
            color=[255, 255, 255],
            type='',
            swap='face-36'),
        72:
        dict(
            name='face-46',
            id=72,
            color=[255, 255, 255],
            type='',
            swap='face-41'),
        73:
        dict(
            name='face-47',
            id=73,
            color=[255, 255, 255],
            type='',
            swap='face-40'),
        74:
        dict(
            name='face-48',
            id=74,
            color=[255, 255, 255],
            type='',
            swap='face-54'),
        75:
        dict(
            name='face-49',
            id=75,
            color=[255, 255, 255],
            type='',
            swap='face-53'),
        76:
        dict(
            name='face-50',
            id=76,
            color=[255, 255, 255],
            type='',
            swap='face-52'),
        77:
        dict(name='face-51', id=77, color=[255, 255, 255], type='', swap=''),
        78:
        dict(
            name='face-52',
            id=78,
            color=[255, 255, 255],
            type='',
            swap='face-50'),
        79:
        dict(
            name='face-53',
            id=79,
            color=[255, 255, 255],
            type='',
            swap='face-49'),
        80:
        dict(
            name='face-54',
            id=80,
            color=[255, 255, 255],
            type='',
            swap='face-48'),
        81:
        dict(
            name='face-55',
            id=81,
            color=[255, 255, 255],
            type='',
            swap='face-59'),
        82:
        dict(
            name='face-56',
            id=82,
            color=[255, 255, 255],
            type='',
            swap='face-58'),
        83:
        dict(name='face-57', id=83, color=[255, 255, 255], type='', swap=''),
        84:
        dict(
            name='face-58',
            id=84,
            color=[255, 255, 255],
            type='',
            swap='face-56'),
        85:
        dict(
            name='face-59',
            id=85,
            color=[255, 255, 255],
            type='',
            swap='face-55'),
        86:
        dict(
            name='face-60',
            id=86,
            color=[255, 255, 255],
            type='',
            swap='face-64'),
        87:
        dict(
            name='face-61',
            id=87,
            color=[255, 255, 255],
            type='',
            swap='face-63'),
        88:
        dict(name='face-62', id=88, color=[255, 255, 255], type='', swap=''),
        89:
        dict(
            name='face-63',
            id=89,
            color=[255, 255, 255],
            type='',
            swap='face-61'),
        90:
        dict(
            name='face-64',
            id=90,
            color=[255, 255, 255],
            type='',
            swap='face-60'),
        91:
        dict(
            name='face-65',
            id=91,
            color=[255, 255, 255],
            type='',
            swap='face-67'),
        92:
        dict(name='face-66', id=92, color=[255, 255, 255], type='', swap=''),
        93:
        dict(
            name='face-67',
            id=93,
            color=[255, 255, 255],
            type='',
            swap='face-65'),
        94:
        dict(
            name='left_hand_root',
            id=94,
            color=[255, 255, 255],
            type='',
            swap='right_hand_root'),
        95:
        dict(
            name='left_thumb1',
            id=95,
            color=[255, 128, 0],
            type='',
            swap='right_thumb1'),
        96:
        dict(
            name='left_thumb2',
            id=96,
            color=[255, 128, 0],
            type='',
            swap='right_thumb2'),
        97:
        dict(
            name='left_thumb3',
            id=97,
            color=[255, 128, 0],
            type='',
            swap='right_thumb3'),
        98:
        dict(
            name='left_thumb4',
            id=98,
            color=[255, 128, 0],
            type='',
            swap='right_thumb4'),
        99:
        dict(
            name='left_forefinger1',
            id=99,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger1'),
        100:
        dict(
            name='left_forefinger2',
            id=100,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger2'),
        101:
        dict(
            name='left_forefinger3',
            id=101,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger3'),
        102:
        dict(
            name='left_forefinger4',
            id=102,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger4'),
        103:
        dict(
            name='left_middle_finger1',
            id=103,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger1'),
        104:
        dict(
            name='left_middle_finger2',
            id=104,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger2'),
        105:
        dict(
            name='left_middle_finger3',
            id=105,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger3'),
        106:
        dict(
            name='left_middle_finger4',
            id=106,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger4'),
        107:
        dict(
            name='left_ring_finger1',
            id=107,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger1'),
        108:
        dict(
            name='left_ring_finger2',
            id=108,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger2'),
        109:
        dict(
            name='left_ring_finger3',
            id=109,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger3'),
        110:
        dict(
            name='left_ring_finger4',
            id=110,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger4'),
        111:
        dict(
            name='left_pinky_finger1',
            id=111,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger1'),
        112:
        dict(
            name='left_pinky_finger2',
            id=112,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger2'),
        113:
        dict(
            name='left_pinky_finger3',
            id=113,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger3'),
        114:
        dict(
            name='left_pinky_finger4',
            id=114,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger4'),
        115:
        dict(
            name='right_hand_root',
            id=115,
            color=[255, 255, 255],
            type='',
            swap='left_hand_root'),
        116:
        dict(
            name='right_thumb1',
            id=116,
            color=[255, 128, 0],
            type='',
            swap='left_thumb1'),
        117:
        dict(
            name='right_thumb2',
            id=117,
            color=[255, 128, 0],
            type='',
            swap='left_thumb2'),
        118:
        dict(
            name='right_thumb3',
            id=118,
            color=[255, 128, 0],
            type='',
            swap='left_thumb3'),
        119:
        dict(
            name='right_thumb4',
            id=119,
            color=[255, 128, 0],
            type='',
            swap='left_thumb4'),
        120:
        dict(
            name='right_forefinger1',
            id=120,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger1'),
        121:
        dict(
            name='right_forefinger2',
            id=121,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger2'),
        122:
        dict(
            name='right_forefinger3',
            id=122,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger3'),
        123:
        dict(
            name='right_forefinger4',
            id=123,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger4'),
        124:
        dict(
            name='right_middle_finger1',
            id=124,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger1'),
        125:
        dict(
            name='right_middle_finger2',
            id=125,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger2'),
        126:
        dict(
            name='right_middle_finger3',
            id=126,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger3'),
        127:
        dict(
            name='right_middle_finger4',
            id=127,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger4'),
        128:
        dict(
            name='right_ring_finger1',
            id=128,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger1'),
        129:
        dict(
            name='right_ring_finger2',
            id=129,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger2'),
        130:
        dict(
            name='right_ring_finger3',
            id=130,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger3'),
        131:
        dict(
            name='right_ring_finger4',
            id=131,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger4'),
        132:
        dict(
            name='right_pinky_finger1',
            id=132,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger1'),
        133:
        dict(
            name='right_pinky_finger2',
            id=133,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger2'),
        134:
        dict(
            name='right_pinky_finger3',
            id=134,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger3'),
        135:
        dict(
            name='right_pinky_finger4',
            id=135,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger4')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_hip', 'hip'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('right_ankle', 'right_knee'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_knee', 'right_hip'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('right_hip', 'hip'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('head', 'neck'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('neck', 'hip'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('neck', 'left_shoulder'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('left_shoulder', 'left_elbow'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('neck', 'right_shoulder'), id=11, color=[255, 128, 0]),
        12:
        dict(
            link=('right_shoulder', 'right_elbow'), id=12, color=[255, 128,
                                                                  0]),
        13:
        dict(link=('right_elbow', 'right_wrist'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('left_eye', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('nose', 'left_eye'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('nose', 'right_eye'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_eye', 'left_ear'), id=17, color=[51, 153, 255]),
        18:
        dict(link=('right_eye', 'right_ear'), id=18, color=[51, 153, 255]),
        19:
        dict(link=('left_ear', 'left_shoulder'), id=19, color=[51, 153, 255]),
        20:
        dict(
            link=('right_ear', 'right_shoulder'), id=20, color=[51, 153, 255]),
        21:
        dict(link=('left_ankle', 'left_big_toe'), id=21, color=[0, 255, 0]),
        22:
        dict(link=('left_ankle', 'left_small_toe'), id=22, color=[0, 255, 0]),
        23:
        dict(link=('left_ankle', 'left_heel'), id=23, color=[0, 255, 0]),
        24:
        dict(
            link=('right_ankle', 'right_big_toe'), id=24, color=[255, 128, 0]),
        25:
        dict(
            link=('right_ankle', 'right_small_toe'),
            id=25,
            color=[255, 128, 0]),
        26:
        dict(link=('right_ankle', 'right_heel'), id=26, color=[255, 128, 0]),
        27:
        dict(link=('left_wrist', 'left_thumb1'), id=27, color=[255, 128, 0]),
        28:
        dict(link=('left_thumb1', 'left_thumb2'), id=28, color=[255, 128, 0]),
        29:
        dict(link=('left_thumb2', 'left_thumb3'), id=29, color=[255, 128, 0]),
        30:
        dict(link=('left_thumb3', 'left_thumb4'), id=30, color=[255, 128, 0]),
        31:
        dict(
            link=('left_wrist', 'left_forefinger1'),
            id=31,
            color=[255, 153, 255]),
        32:
        dict(
            link=('left_forefinger1', 'left_forefinger2'),
            id=32,
            color=[255, 153, 255]),
        33:
        dict(
            link=('left_forefinger2', 'left_forefinger3'),
            id=33,
            color=[255, 153, 255]),
        34:
        dict(
            link=('left_forefinger3', 'left_forefinger4'),
            id=34,
            color=[255, 153, 255]),
        35:
        dict(
            link=('left_wrist', 'left_middle_finger1'),
            id=35,
            color=[102, 178, 255]),
        36:
        dict(
            link=('left_middle_finger1', 'left_middle_finger2'),
            id=36,
            color=[102, 178, 255]),
        37:
        dict(
            link=('left_middle_finger2', 'left_middle_finger3'),
            id=37,
            color=[102, 178, 255]),
        38:
        dict(
            link=('left_middle_finger3', 'left_middle_finger4'),
            id=38,
            color=[102, 178, 255]),
        39:
        dict(
            link=('left_wrist', 'left_ring_finger1'),
            id=39,
            color=[255, 51, 51]),
        40:
        dict(
            link=('left_ring_finger1', 'left_ring_finger2'),
            id=40,
            color=[255, 51, 51]),
        41:
        dict(
            link=('left_ring_finger2', 'left_ring_finger3'),
            id=41,
            color=[255, 51, 51]),
        42:
        dict(
            link=('left_ring_finger3', 'left_ring_finger4'),
            id=42,
            color=[255, 51, 51]),
        43:
        dict(
            link=('left_wrist', 'left_pinky_finger1'),
            id=43,
            color=[0, 255, 0]),
        44:
        dict(
            link=('left_pinky_finger1', 'left_pinky_finger2'),
            id=44,
            color=[0, 255, 0]),
        45:
        dict(
            link=('left_pinky_finger2', 'left_pinky_finger3'),
            id=45,
            color=[0, 255, 0]),
        46:
        dict(
            link=('left_pinky_finger3', 'left_pinky_finger4'),
            id=46,
            color=[0, 255, 0]),
        47:
        dict(link=('right_wrist', 'right_thumb1'), id=47, color=[255, 128, 0]),
        48:
        dict(
            link=('right_thumb1', 'right_thumb2'), id=48, color=[255, 128, 0]),
        49:
        dict(
            link=('right_thumb2', 'right_thumb3'), id=49, color=[255, 128, 0]),
        50:
        dict(
            link=('right_thumb3', 'right_thumb4'), id=50, color=[255, 128, 0]),
        51:
        dict(
            link=('right_wrist', 'right_forefinger1'),
            id=51,
            color=[255, 153, 255]),
        52:
        dict(
            link=('right_forefinger1', 'right_forefinger2'),
            id=52,
            color=[255, 153, 255]),
        53:
        dict(
            link=('right_forefinger2', 'right_forefinger3'),
            id=53,
            color=[255, 153, 255]),
        54:
        dict(
            link=('right_forefinger3', 'right_forefinger4'),
            id=54,
            color=[255, 153, 255]),
        55:
        dict(
            link=('right_wrist', 'right_middle_finger1'),
            id=55,
            color=[102, 178, 255]),
        56:
        dict(
            link=('right_middle_finger1', 'right_middle_finger2'),
            id=56,
            color=[102, 178, 255]),
        57:
        dict(
            link=('right_middle_finger2', 'right_middle_finger3'),
            id=57,
            color=[102, 178, 255]),
        58:
        dict(
            link=('right_middle_finger3', 'right_middle_finger4'),
            id=58,
            color=[102, 178, 255]),
        59:
        dict(
            link=('right_wrist', 'right_ring_finger1'),
            id=59,
            color=[255, 51, 51]),
        60:
        dict(
            link=('right_ring_finger1', 'right_ring_finger2'),
            id=60,
            color=[255, 51, 51]),
        61:
        dict(
            link=('right_ring_finger2', 'right_ring_finger3'),
            id=61,
            color=[255, 51, 51]),
        62:
        dict(
            link=('right_ring_finger3', 'right_ring_finger4'),
            id=62,
            color=[255, 51, 51]),
        63:
        dict(
            link=('right_wrist', 'right_pinky_finger1'),
            id=63,
            color=[0, 255, 0]),
        64:
        dict(
            link=('right_pinky_finger1', 'right_pinky_finger2'),
            id=64,
            color=[0, 255, 0]),
        65:
        dict(
            link=('right_pinky_finger2', 'right_pinky_finger3'),
            id=65,
            color=[0, 255, 0]),
        66:
        dict(
            link=('right_pinky_finger3', 'right_pinky_finger4'),
            id=66,
            color=[0, 255, 0])
    },
    joint_weights=[1.] * 136,

    # 'https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/'
    # 'HalpeCOCOAPI/PythonAPI/halpecocotools/cocoeval.py#L245'
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.08, 0.08, 0.08,
        0.089, 0.089, 0.089, 0.089, 0.089, 0.089, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.015, 0.015, 0.015, 0.015, 0.015, 0.015
    ])
