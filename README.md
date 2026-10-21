# 水下珊瑚识别与 XYZ 测距

普通 USB 左右拼接双目相机的实时珊瑚检测、图像增强和水下标定测距程序。使用 uv 管理 Python 和依赖。

## 安装与启动

- Windows 64 位；相机输出 `2560×720`，左右各 `1280×720`，顺序为 `LEFT|RIGHT`。
- uv 至少为 `0.9.8`；当前环境使用 `0.13.4`。uv 自动准备 Python `3.9.23`，无需 Anaconda。
- 首次安装需要联网，并建议预留至少 12 GB 空间。PyTorch 下载约 2.3 GiB。

在项目目录打开 PowerShell：

```powershell
uv --version
uv sync --locked
uv run --locked python run.py
```

uv 太旧时按原安装方式更新；通过官方独立安装器安装的 uv 可以使用 `uv self update`。
本项目的依赖排除设置从 [uv 0.9.8](https://github.com/astral-sh/uv/releases/tag/0.9.8) 开始支持。

默认相机编号为 `1`。编号不同或需要指定计算设备时：

```powershell
# 指定相机，使用 CPU（当前 AMD 显卡电脑适用）
uv run --locked python run.py --camera 1 --device cpu

# 第一块与当前 PyTorch/CUDA 版本兼容的 NVIDIA GPU
uv run --locked python run.py --camera 1 --device 0

# 关闭画面增强
uv run --locked python run.py --input-enhancement off

# 查看参数
uv run --locked python run.py --help
```

默认自动选择可用的 NVIDIA CUDA，否则使用 CPU。当前版本没有 AMD DirectML 推理后端，
不能通过 `--device 0` 启用 RX 7800 XT。CUDA 运行库随 PyTorch 安装，无需另装 CUDA Toolkit；
Windows VC++ 运行库及 NVIDIA 驱动属于系统组件。

## 窗口与录制

点击预览窗口使其获得键盘焦点：

| 操作 | 效果 |
|---|---|
| `Q` | 开始录制 |
| `W` 或 `Esc` | 保存录像并退出 |
| 拖动窗口四边或四角 | 宽高按画布比例联动缩放 |
| 最大化或贴靠窗口 | 画面等比例显示，空余区域留边 |

左侧显示增强彩色画面、检测框、类别、置信度和 XYZ，右侧显示深度图与色标。
每个图像面板为 `16:9`；包含左右面板、文字和色标的完整画布为 `2688×840`（`16:5`）。
缩放限制作用于显示客户区，并考虑标题栏和边框；预览缩放不改变标定、计算或录像尺寸。

每次运行输出到 `coral_runs/日期时间/`：

| 文件 | 内容 |
|---|---|
| `visualization.mp4` | 按 Q 后保存的完整画面；MP4 编码不可用时回退到 AVI |
| `detections.jsonl` | 检测类别、置信度、框坐标及有效 XYZ |
| `frame_metrics.csv` | 采集 FPS、实际处理 FPS、帧龄及各阶段耗时 |
| `settings.json`、`camera.json` | 本次参数、计算设备与实际相机模式 |
| `summary.json`、`recording.json` | 处理与录像统计 |
| `calibration_snapshot.json` | 本次使用的标定快照 |

视频容器的 30 FPS 包含保持上一帧的编码，**不代表每秒有 30 个新识别结果**。

## 相机、标定与 XYZ

当前启动配置对应 2026-10-13 的 USB 彩色双目水下标定，不是 D435i 的 `640×480` 标定。

- 采集：DirectShow / MJPG，目标 30 FPS，实际模式记录在 `camera.json`。
- 标定：`config/stereo_opencv.yaml`；对应 JSON 为 `config/calibration.json`。
- 基线：`59.55734255 mm`；不能随意交换左右目。
- 原始左目：`fx=1406.8097219, fy=1409.6292648, cx=634.4973071, cy=490.6416794`。
- 原始右目：`fx=1411.7321975, fy=1413.6009901, cx=686.5858993, cy=483.0347250`。
- 校正后 P1：`fx=fy=1688.2932880, cx=797.5483360, cy=471.8612728`。

XYZ 使用校正后的像素坐标和 P1，不能把原始 K1 直接套到校正图上。
坐标原点为校正后的左相机光心，X 向右、Y 向下、Z 向前；内部及日志单位为 m，画面标签单位为 cm：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

目标 XYZ 是检测框中心区域有效三维点的稳健中位数；Z 表示前向深度，不是直线距离。
原标定保留 `quality_passed=false`，距离精度仍需用已知距离实测确认。

## 当前处理配置与性能

默认使用 mild 增强、YOLO `imgsz=416`、置信度至少 `0.60`、OpenCV 2 线程。
CUDA 模式默认 native 卷积和 FP16；CPU 模式自动关闭 FP16。
双目匹配尺寸为 `320×180`，检测与输出视野保持每目 `1280×720`。

增强只用于检测输入和左侧显示；原始左右目继续用于双目匹配与几何测量。
深度补洞只影响显示，XYZ 使用严格有效匹配点。当前检测和深度仍逐帧同步输出。

2026-10-21，本机一次 CPU 实机运行（4189 帧、约 286 秒，录制开启）的基线：

| 指标 | 结果 |
|---|---:|
| 实际采集到达速度 | 平均约 17.9 FPS |
| 整段平均处理速度 | 约 14.6 FPS |
| 双目处理 | 平均约 52 ms |
| YOLO 推理 | 平均约 21 ms |
| 图像增强 | 平均约 9 ms |

上述处理有并行，耗时不能直接相加。帧龄是主机收帧后的统计，不是完整的相机到控制器延迟。
这些数据用于比较后续改动，不保证不同场景或电脑上的速度；当前没有新增 GPU 加速或跟踪控制接口。

## 代码与依赖

| 文件 | 职责 |
|---|---|
| `run.py` | 实时启动入口 |
| `coral_live.py` | 检测、标定几何、深度、绘图、日志与录像主流程 |
| `camera.py`、`capture_process.py` | 相机读取与独立进程采集 |
| `preview_window.py` | Windows 预览窗口比例约束与等比例显示 |
| `guided_depth_gpu.py`、`cuda_guided_kernel.py` | CUDA 深度重建 |
| `cpu_depth_display.py`、`image_enhancement.py` | CPU 深度显示补洞与画面增强 |
| `capture_quality.py`、`imaging.py` | 采集质量及图像辅助逻辑 |
| `rgb_depth_projection.py` | RealSense RGB 投影辅助逻辑 |
| `models/best.pt`、`config/` | 模型和对应标定 |
| `pyproject.toml`、`uv.lock`、`.python-version` | 项目设置、依赖锁和 Python 版本 |
| `MANIFEST.sha256` | 当前版本文件校验清单 |

核心依赖为 torch `1.13.1+cu116`、torchvision `0.14.1+cu116`、ultralytics `8.3.199`、
numpy `1.26.4`、opencv-contrib-python `4.13.0.92`。

项目刻意固定 NumPy 1.x，并排除 `opencv-python`，由 contrib 唯一提供 `cv2` 与 `ximgproc`。
普通依赖元数据检查可能提示 OpenCV 的 NumPy 要求或 Ultralytics 的 `opencv-python` 缺失；
这里使用的是已运行验证的覆盖配置，不要因此另装第二套 OpenCV 或升级 NumPy。
安装后继续使用 `uv sync --locked`，保持锁定版本。

## Git 与验证

代码、模型、标定和依赖锁纳入 Git；`.venv/`、`coral_runs/`、缓存和运行日志由 `.gitignore` 排除。
`.gitattributes` 固定文本为 LF，避免跨 Windows 检出时换行符改变文件校验值。

```powershell
git status
git log --oneline

# 检查四边/四角缩放、最小尺寸、最大化和关闭；不会打开相机
uv run --locked python test_preview_window.py

# 离线运行已有的原始左右图片，测试计算与编码，不会打开相机
uv run --locked python coral_live.py --offline "D:\datasets\stereo" --frames 30 --device cpu --record-offline
```

离线目录需包含 `original/*_left.png` 和对应的 `*_right.png`，每张图片为 `1280×720`。
需要同时验证预览时追加 `--show-offline`。
