# 3D-Box via Segment Anything

We extend [Segment Anything](https://github.com/facebookresearch/segment-anything) to 3D perception by combining it with [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt). Note that this project is still in progress. We are improving it and developing more examples. Any issue or pull request is welcome!

<p align="center"> <img src="images/sam-voxelnext.png" width="100%"> </p>

## Why this project?
[Segment Anything](https://github.com/facebookresearch/segment-anything) and its following projects
focus on 2D images. In this project, we extend the scope to 3D world by combining [Segment Anything](https://github.com/facebookresearch/segment-anything) and [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt). When we provide a prompt (e.g., a point / box), the result is not only 2D segmentation mask, but also 3D boxes.

The core idea is that [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt) is a fully sparse 3D detector. It predicts 3D object upon each sparse voxel. We project 3D sparse voxels onto 2D images. And then 3D boxes can be generated for voxels in the SAM mask. 

- This project makes 3D object detection to be promptable.
- VoxelNeXt is based on sparse voxels that are easy to be related to the mask generated from segment anything.
- This project could facilitate 3D box labeling. 3D box can be obtained via a simple click on image. It might largely save human efforts, especially on autonuous driving scenes.

## Installation
1. Basic requirements
`pip install -r requirements.txt
`
2. Segment anything
`pip install git+https://github.com/facebookresearch/segment-anything.git
`
3. spconv
`pip install spconv
`
or cuda version spconv `pip install spconv-cu111` based on your cuda version. Please use spconv 2.2 / 2.3 version, for example spconv==2.3.5


## Getting Started
Please try it via [seg_anything_and_3D.ipynb](seg_anything_and_3D.ipynb).
We provide this example on nuScenes dataset. You can use other image-points pairs. 

- The demo point for one frame is provided here [points_demo.npy](https://drive.google.com/file/d/1br0VDamameu7B1G1p4HEjj6LshGs5dHB/view?usp=share_link).
- The point to image translation infos on nuScenes val can be download [here](https://drive.google.com/file/d/1nJqdfs0gMTIo4fjOwytSbM0fdBOJ4IGb/view?usp=share_link).
- The weight in the demo is [voxelnext_nuscenes_kernel1.pth](https://drive.google.com/file/d/17mQRXXUsaD0dlRzAKep3MQjfj8ugDsp9/view?usp=share_link).
- The nuScenes info file is [nuscenes_infos_10sweeps_val.pkl](https://drive.google.com/file/d/1Kaxtubzr1GofcoFz97S6qwAIG2wzhPo_/view?usp=share_link). This is generated from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) codebase.


<p align="center"> <img src="images/mask_box.png" width="100%"> </p>
<p align="center"> <img src="images/image_boxes1.png" width="100%"> </p>
<p align="center"> <img src="images/image_boxes2.png" width="100%"> </p>
<p align="center"> <img src="images/image_boxes3.png" width="100%"> </p>

## TODO List
- - [ ] Zero-shot version VoxelNeXt.
- - [ ] Examples on more datasets.
- - [ ] Indoor scenes.

## Citation 
If you find this project useful in your research, please consider citing:
```
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{chen2023voxenext,
  title={VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking},
  author={Yukang Chen and Jianhui Liu and Xiangyu Zhang and Xiaojuan Qi and Jiaya Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

```

## Acknowledgement
-  [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)
- [UVTR](https://github.com/dvlab-research/UVTR) for 3D to 2D translation.
