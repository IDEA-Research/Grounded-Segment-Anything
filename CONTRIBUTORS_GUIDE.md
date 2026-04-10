# 项目交接与协作说明

## 一、研发模式声明

本项目采用 **Tech Lead 统一推码** 模式。

- 底层 AI 链路、前端 UI、后端逻辑、推理优化、Bug 修复，统一由 Tech Lead 配合 Codex 完成。
- 团队其他成员 **不直接修改代码**，尤其严禁修改 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py) 及任何核心脚本。
- 所有功能迭代、界面优化、文案调整、交互建议，统一先沉淀为需求文档、草图、示意图或竞品分析，再交由 Tech Lead 使用高阶 AI 统一生成并实装。

### 严禁直接修改的核心文件

- [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py)
- [requirements.txt](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\requirements.txt)
- [scripts/check_blackwell_cuda.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\scripts\check_blackwell_cuda.py)
- `GroundingDINO/`
- `segment_anything/`
- `EfficientSAM/`

### 代码区块定位说明

以下内容仅用于团队理解系统结构，**不是授权修改**：

- CSS 样式配置位于 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py):85 左右的 `APP_CSS`
- SAM 组件与主模型调度位于 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py):686 和 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py):718 附近
- 主推理入口位于 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py):960 左右的 `run_grounded_sam`
- 前端异常透传入口位于 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py):1125 左右的 `run_grounded_sam_ui`
- Gradio SaaS 布局区位于 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py):1198 之后，重点包括 `gr.Blocks(css=APP_CSS, ...)`、`with gr.Row(...)` 等区块

结论：团队成员可以阅读结构帮助理解产品，但 **不要直接改这些代码块**。

## 二、4 大纯业务角色分工

### 1. 产品视觉设计师 Product Designer

**职责**

- 不写代码
- 对标 `Photoroom`、`Canva`、`Clipdrop` 等图像 SaaS 产品，调研首页风格、控制台布局、按钮层级、状态反馈、加载动效与提示文案
- 将 UI 升级诉求整理成结构化需求单，方便 Tech Lead 直接交给 AI 生成

**交付内容**

- 竞品截图与分析
- 色板建议、字体建议、按钮排版建议
- 页面线框图、模块说明、交互动线
- 文案替换建议，例如上传区提示、处理中提示、异常提示

**文件领地**

- [docs/ui_design/README.md](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\docs\ui_design\README.md)
- `docs/ui_design/` 下新增的 Markdown、图片、草图、参考截图

**协作方式**

- 只提交设计图、原型图、需求文档
- 若想新增复杂交互，不自行写代码，而是把需求拆成“目标效果 + 页面位置 + 交互说明 + 参考图”四部分提交给 Tech Lead

### 2. QA 测试与数据集负责人

**职责**

- 不写代码
- 收集 50 到 100 张真实测试图，覆盖电商白底图、复杂商品图、老照片、人像、遮挡目标、低清噪声图等场景
- 记录成功案例、失败案例、边界案例，为答辩准备可展示的效果图鉴

**交付内容**

- 原始测试图
- 处理后效果图
- Before / After 对照图
- 测试记录表，至少包含：图片类型、使用模式、提示词、结果评价、失败原因

**文件领地**

- [dataset/README.md](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\dataset\README.md)
- `dataset/raw/`
- `dataset/before_after/`
- `dataset/failure_cases/`
- 可复用现有 [assets](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\assets) 作为演示样例参考，但不改代码

**协作方式**

- 只上传素材与 Markdown 记录
- 不进入任何 Python 文件，不改任何依赖和脚本

### 3. 商业企划与 PPT 架构师

**职责**

- 不写代码
- 提炼本项目的技术壁垒、产品价值与应用场景
- 撰写大创申报书
- 组织 10 到 15 页路演 PPT 内容

**建议重点提炼的技术亮点**

- `PyTorch Nightly` 打通 `RTX 5060 / sm_120`
- 8GB 显存下的半精度与 CPU Offload 调度
- GroundingDINO 双后端兜底方案
- 中文提示词轻量翻译与后端异常透传
- 面向电商修图和老照片修复的 SaaS 化工作台

**文件领地**

- `docs/`
- 建议重点维护：
  - `docs/pitch/`
  - `docs/plan/`
  - `docs/qa/`

**协作方式**

- 所有内容以 Markdown、PPT、图片、表格形式沉淀
- 需要技术细节时，引用本仓库已有说明或向 Tech Lead 提问，不自行修改代码描述

### 4. 路演主讲人 Speaker

**职责**

- 不写代码
- 熟悉产品演示流程
- 撰写 5 到 8 分钟路演逐字稿
- 梳理评委高频问题并准备答辩话术

**建议准备的材料**

- 演示步骤卡
- 逐页讲稿
- Q&A 题库
- 现场异常应对话术，例如网络波动、模型加载等待、样例切换

**文件领地**

- [README.md](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\README.md)
- `docs/`
- 建议新增：
  - `docs/demo_script/`
  - `docs/qa_bank/`

**协作方式**

- 可更新功能介绍、演示说明、答辩文案
- 不修改功能实现代码

## 三、协作规范

团队成员只需要通过以下两类方式协作：

- 在本地上传图片素材、对比图、Markdown 文档
- 在 GitHub 网页端或本地提交 Pull Request

### 允许提交的内容

- 图片素材
- 对比图
- 需求文档
- UI 草图
- PPT 文案
- Q&A 话术稿

### 禁止提交的内容

- 对 [gradio_app.py](C:\Users\Lenovo\Desktop\Grounded-Segment\Grounded-Segment-Anything\gradio_app.py) 的任何代码修改
- 对 `requirements.txt`、模型脚本、推理逻辑、依赖环境的任何改动
- 自行新增不明脚本或替换核心模型文件

### 推荐协作流程

1. 先产出素材或文档
2. 提交到 `docs/`、`dataset/`、`assets/` 等非代码区域
3. 在 PR 或微信群中说明“目标效果、使用场景、期望改动”
4. 由 Tech Lead 统一评估并决定是否进入代码实现

## 四、给全队的一句话

这阶段大家最重要的任务，不是和底层代码硬碰硬，而是一起把产品价值、测试样本、演示效果和商业表达做扎实。代码侧由 Tech Lead 统一兜底，大家把业务包装做到位，我们答辩会更稳。
