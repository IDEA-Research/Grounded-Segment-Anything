<!-- [OTHERS] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2112.13715">SmoothNet (arXiv'2021)</a></summary>

```bibtex
@article{zeng2021smoothnet,
  title={SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos},
  author={Zeng, Ailing and Yang, Lei and Ju, Xuan and Li, Jiefeng and Wang, Jianyi and Xu, Qiang},
  journal={arXiv preprint arXiv:2112.13715},
  year={2021}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>

The following SmoothNet model checkpoints are available for pose smoothing. The table shows the the performance of [SimpleBaseline3D](https://arxiv.org/abs/1705.03098) on [Human3.6M](https://ieeexplore.ieee.org/abstract/document/6682899/) dataset without/with the SmoothNet plugin, and compares the SmoothNet models with 4 different window sizes (8, 16, 32 and 64). The metrics are MPJPE(mm), P-MEJPE(mm) and Acceleration Error (mm/frame^2).

| Arch                                 | Window Size | MPJPE<sup>w/o</sup> | MPJPE<sup>w</sup> | P-MPJPE<sup>w/o</sup> | P-MPJPE<sup>w</sup> | AC. Err<sup>w/o</sup> | AC. Err<sup>w</sup> |                 ckpt                  |
| :----------------------------------- | :---------: | :-----------------: | :---------------: | :-------------------: | :-----------------: | :-------------------: | :-----------------: | :-----------------------------------: |
| [smoothnet_ws8](/configs/_base_/filters/smoothnet_t8_h36m.py) |      8      |        54.48        |       53.15       |         42.20         |        41.32        |         19.18         |        1.87         | [ckpt](https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws8_h36m.pth) |
| [smoothnet_ws16](/configs/_base_/filters/smoothnet_t16_h36m.py) |     16      |        54.48        |       52.74       |         42.20         |        41.20        |         19.18         |        1.22         | [ckpt](https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws16_h36m.pth) |
| [smoothnet_ws32](/configs/_base_/filters/smoothnet_t32_h36m.py) |     32      |        54.48        |       52.47       |         42.20         |        40.84        |         19.18         |        0.99         | [ckpt](https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws32_h36m.pth) |
| [smoothnet_ws64](/configs/_base_/filters/smoothnet_t64_h36m.py) |     64      |        54.48        |       53.37       |         42.20         |        40.77        |         19.18         |        0.92         | [ckpt](https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws64_h36m.pth) |
