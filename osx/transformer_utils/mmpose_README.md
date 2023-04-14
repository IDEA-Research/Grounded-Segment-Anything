<div align="center">
  <img src="resources/mmpose-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b>OpenMMLab website</b>
    <sup>
      <a href="https://openmmlab.com">
        <i>HOT</i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>OpenMMLab platform</b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i>TRY IT OUT</i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://img.shields.io/pypi/v/mmpose)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

[📘Documentation](https://mmpose.readthedocs.io/en/v0.28.0/) |
[🛠️Installation](https://mmpose.readthedocs.io/en/v0.28.0/install.html) |
[👀Model Zoo](https://mmpose.readthedocs.io/en/v0.28.0/modelzoo.html) |
[📜Papers](https://mmpose.readthedocs.io/en/v0.28.0/papers/algorithms.html) |
[🆕Update News](https://mmpose.readthedocs.io/en/v0.28.0/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmpose/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_CN.md)

</div>

## Introduction

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.5+**.

https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4

<details open>
<summary><b>Major Features</b></summary>

- **Support diverse tasks**

  We support a wide spectrum of mainstream pose analysis tasks in current research community, including 2d multi-person human pose estimation, 2d hand pose estimation, 2d face landmark detection, 133 keypoint whole-body human pose estimation, 3d human mesh recovery, fashion landmark detection and animal pose estimation.
  See [demo.md](demo/README.md) for more information.

- **Higher efficiency and higher accuracy**

  MMPose implements multiple state-of-the-art (SOTA) deep learning models, including both top-down & bottom-up approaches. We achieve faster training speed and higher accuracy than other popular codebases, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
  See [benchmark.md](docs/en/benchmark.md) for more information.

- **Support for various datasets**

  The toolbox directly supports multiple popular and representative datasets, COCO, AIC, MPII, MPII-TRB, OCHuman etc.
  See [data_preparation.md](docs/en/data_preparation.md) for more information.

- **Well designed, tested and documented**

  We decompose MMPose into different components and one can easily construct a customized
  pose estimation framework by combining different modules.
  We provide detailed documentation and API reference, as well as unittests.

</details>

## What's New

- 2022-07-06: MMPose [v0.28.0](https://github.com/open-mmlab/mmpose/releases/tag/v0.28.0) is released. Major updates include:
  - Support [TCFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zeng_Not_All_Tokens_Are_Equal_Human-Centric_Visual_Analysis_via_Token_CVPR_2022_paper.html) (CVPR'2022). See the [model page](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_coco-wholebody.md)
  - Add [RLE](https://arxiv.org/abs/2107.11291) pre-trained model on COCO dataset. See the [model page](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/resnet_rle_coco.md)
  - Update [Swin](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_coco.md) models with better performance
- 2022-02-28: MMPose model deployment is supported by [MMDeploy](https://github.com/open-mmlab/mmdeploy) v0.3.0
  MMPose Webcam API is a simple yet powerful tool to develop interactive webcam applications with MMPose features.
- 2021-12-29: OpenMMLab Open Platform is online! Try our [pose estimation demo](https://platform.openmmlab.com/web-demo/demo/poseestimation)

## Installation

MMPose depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.
Please refer to [install.md](docs/en/install.md) for detailed installation guide.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip3 install -e .
```

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMPose.
There are also tutorials:

- [learn about configs](docs/en/tutorials/0_config.md)
- [finetune model](docs/en/tutorials/1_finetune.md)
- [add new dataset](docs/en/tutorials/2_new_dataset.md)
- [customize data pipelines](docs/en/tutorials/3_data_pipeline.md)
- [add new modules](docs/en/tutorials/4_new_modules.md)
- [export a model to ONNX](docs/en/tutorials/5_export_model.md)
- [customize runtime settings](docs/en/tutorials/6_customize_runtime.md)

## Model Zoo

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html) page.

<details open>
<summary><b>Supported algorithms:</b></summary>

- [x] [DeepPose](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#deeppose-cvpr-2014) (CVPR'2014)
- [x] [CPM](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#cpm-cvpr-2016) (CVPR'2016)
- [x] [Hourglass](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hourglass-eccv-2016) (ECCV'2016)
- [x] [SimpleBaseline3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#simplebaseline3d-iccv-2017) (ICCV'2017)
- [x] [Associative Embedding](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#associative-embedding-nips-2017) (NeurIPS'2017)
- [x] [HMR](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#hmr-cvpr-2018) (CVPR'2018)
- [x] [SimpleBaseline2D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#simplebaseline2d-eccv-2018) (ECCV'2018)
- [x] [HRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019) (CVPR'2019)
- [x] [VideoPose3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#videopose3d-cvpr-2019) (CVPR'2019)
- [x] [HRNetv2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnetv2-tpami-2019) (TPAMI'2019)
- [x] [MSPN](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mspn-arxiv-2019) (ArXiv'2019)
- [x] [SCNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#scnet-cvpr-2020) (CVPR'2020)
- [x] [HigherHRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#higherhrnet-cvpr-2020) (CVPR'2020)
- [x] [RSN](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#rsn-eccv-2020) (ECCV'2020)
- [x] [InterNet](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#internet-eccv-2020) (ECCV'2020)
- [x] [VoxelPose](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#voxelpose-eccv-2020) (ECCV'2020)
- [x] [LiteHRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#litehrnet-cvpr-2021) (CVPR'2021)
- [x] [ViPNAS](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#vipnas-cvpr-2021) (CVPR'2021)

</details>

<details open>
<summary><b>Supported techniques:</b></summary>

- [x] [FPN](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#fpn-cvpr-2017) (CVPR'2017)
- [x] [FP16](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#fp16-arxiv-2017) (ArXiv'2017)
- [x] [Wingloss](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#wingloss-cvpr-2018) (CVPR'2018)
- [x] [AdaptiveWingloss](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#adaptivewingloss-iccv-2019) (ICCV'2019)
- [x] [DarkPose](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#darkpose-cvpr-2020) (CVPR'2020)
- [x] [UDP](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#udp-cvpr-2020) (CVPR'2020)
- [x] [Albumentations](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#albumentations-information-2020) (Information'2020)
- [x] [SoftWingloss](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#softwingloss-tip-2021) (TIP'2021)
- [x] [SmoothNet](/configs/_base_/filters/smoothnet_h36m.md) (arXiv'2021)
- [x] [RLE](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#rle-iccv-2021) (ICCV'2021)

</details>

<details open>
<summary><b>Supported <a href="https://mmpose.readthedocs.io/en/latest/datasets.html">datasets</a>:</b></summary>

- [x] [AFLW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#aflw-iccvw-2011) \[[homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#jhmdb-iccv-2013) \[[homepage](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [COFW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#cofw-iccv-2013) \[[homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [MPII](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#mpii-cvpr-2014) \[[homepage](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [Human3.6M](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#human3-6m-tpami-2014) \[[homepage](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [COCO](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#coco-eccv-2014) \[[homepage](http://cocodataset.org/)\] (ECCV'2014)
- [x] [CMU Panoptic](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#cmu-panoptic-iccv-2015) \[[homepage](http://domedb.perception.cs.cmu.edu/)\] (ICCV'2015)
- [x] [DeepFashion](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#deepfashion-cvpr-2016) \[[homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [300W](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#300w-imavis-2016) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [RHD](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#rhd-iccv-2017) \[[homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic HandDB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#cmu-panoptic-handdb-cvpr-2017) \[[homepage](http://domedb.perception.cs.cmu.edu/handdb.html)\] (CVPR'2017)
- [x] [AI Challenger](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#ai-challenger-arxiv-2017) \[[homepage](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [MHP](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#mhp-acm-mm-2018) \[[homepage](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [WFLW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#wflw-cvpr-2018) \[[homepage](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [PoseTrack18](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#posetrack18-cvpr-2018) \[[homepage](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [OCHuman](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#ochuman-cvpr-2019) \[[homepage](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#crowdpose-cvpr-2019) \[[homepage](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [MPII-TRB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#mpii-trb-iccv-2019) \[[homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [FreiHand](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#freihand-iccv-2019) \[[homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [Animal-Pose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#animal-pose-iccv-2019) \[[homepage](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [OneHand10K](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#onehand10k-tcsvt-2019) \[[homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [Vinegar Fly](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#vinegar-fly-nature-methods-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#desert-locust-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#grevys-zebra-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#atrw-acm-mm-2020) \[[homepage](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)
- [x] [Halpe](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#halpe-cvpr-2020) \[[homepage](https://github.com/Fang-Haoshu/Halpe-FullBody/)\] (CVPR'2020)
- [x] [COCO-WholeBody](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#coco-wholebody-eccv-2020) \[[homepage](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [MacaquePose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#macaquepose-biorxiv-2020) \[[homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [InterHand2.6M](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#interhand2-6m-eccv-2020) \[[homepage](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [AP-10K](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#ap-10k-neurips-2021) \[[homepage](https://github.com/AlexTheBad/AP-10K)\] (NeurIPS'2021)
- [x] [Horse-10](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#horse-10-wacv-2021) \[[homepage](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)

</details>

<details open>
<summary><b>Supported backbones:</b></summary>

- [x] [AlexNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#alexnet-neurips-2012) (NeurIPS'2012)
- [x] [VGG](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#vgg-iclr-2015) (ICLR'2015)
- [x] [ResNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnet-cvpr-2016) (CVPR'2016)
- [x] [ResNext](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnext-cvpr-2017) (CVPR'2017)
- [x] [SEResNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#seresnet-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV1](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#shufflenetv1-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#shufflenetv2-eccv-2018) (ECCV'2018)
- [x] [MobilenetV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mobilenetv2-cvpr-2018) (CVPR'2018)
- [x] [ResNetV1D](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnetv1d-cvpr-2019) (CVPR'2019)
- [x] [ResNeSt](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#resnest-arxiv-2020) (ArXiv'2020)
- [x] [Swin](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#swin-cvpr-2021) (CVPR'2021)
- [x] [HRFormer](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrformer-nips-2021) (NIPS'2021)
- [x] [PVT](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#pvt-iccv-2021) (ICCV'2021)
- [x] [PVTV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#pvtv2-cvmj-2022) (CVMJ'2022)

</details>

### Model Request

We will keep up with the latest progress of the community, and support more popular algorithms and frameworks. If you have any feature requests, please feel free to leave a comment in [MMPose Roadmap](https://github.com/open-mmlab/mmpose/issues/9).

### Benchmark

#### Accuracy and Training Speed

MMPose achieves superior of training speed and accuracy on the standard keypoint detection benchmarks like COCO. See more details at [benchmark.md](docs/en/benchmark.md).

#### Inference Speed

We summarize the model complexity and inference speed of major models in MMPose, including FLOPs, parameter counts and inference speeds on both CPU and GPU devices with different batch sizes. Please refer to [inference_speed_summary.md](docs/en/inference_speed_summary.md) for more details.

## Data Preparation

Please refer to [data_preparation.md](docs/en/data_preparation.md) for a general knowledge of data preparation.

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMPose. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMPose is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
