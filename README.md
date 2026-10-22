# 水下珊瑚识别与 XYZ 测距

普通 USB 左右拼接双目相机的实时珊瑚检测、图像增强和水下标定测距程序。使用 uv 管理 Python 和依赖。

后续开发见 [珊瑚 yaw 视觉闭环计划](PLAN.md)：已确认网线传视频、USB 直连 Teensy 的双线部署和方案 A（视觉偏差 → 目标航向变化速度 → Teensy 更新目标 yaw → 原有双环），尚未实现。

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

# 默认深度异步不限频；仍可手动限制深度更新频率
uv run --locked python run.py --camera 1 --device cpu --depth-fps 10

# 每帧等待同帧深度（不同于默认的异步不限频）
uv run --locked python run.py --camera 1 --device cpu --depth-fps 0

# 只测 USB 采集：不加载 YOLO，不做深度、预览或视频编码
uv run --locked python run.py --capture-only --camera 1 --frames 150

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

默认只显示左目增强彩色画面、检测框、类别和置信度，右目继续在后台参与异步测距。
识别尺寸保持 `--imgsz 416`，深度匹配保持 `--stereo-width 320`；相机每目仍为 `1280×720`。
顶部显示检测 FPS、当前检测源帧年龄和录像状态；底部显示最新深度源帧编号、年龄和最多两个目标的 XYZ（cm）。
`frame age` 与深度 `age` 均从主机收到各自源帧时计时，不包含相机内部缓冲，也不是完整控制延迟。

目前没有跨帧目标锁定。底部 D1/D2 仅为深度历史帧内的编号，不对应当前彩色画面的目标编号，
也不是持续跟踪 ID；不会把旧 XYZ 贴到当前检测框上。更多目标保留在日志。
没有有效测距点显示 `--`，不使用显示补洞值伪造测距；XYZ 仍只使用严格有效匹配点。
深度尚未就绪显示 WAIT；超过 `--depth-max-age`（默认 0.75 秒）显示 STALE，并隐藏过期 XYZ。
底部保留标定状态、坐标方向和快捷键；计算耗时、有效点数等详细信息见运行日志。

需要查看深度伪彩图、D1/D2 测量框和 cm 色标时，启动时增加调试参数：

```powershell
uv run --locked python run.py --camera 0 --device cpu --show-depth
```

左目图像始终为 `1280×720`（`16:9`），含上下信息栏的默认画布为 `1280×880`；
启用深度调试面板后为 `2688×880`。窗口始终按所选完整画布的比例缩放，图像不会拉伸。
缩放限制考虑标题栏和边框；预览缩放不改变标定、计算或录像尺寸。
隐藏深度面板只改变显示，后台深度算法与异步调度不变。

每次运行输出到 `coral_runs/日期时间/`：

| 文件 | 内容 |
|---|---|
| `visualization.mp4` | 按 Q 后保存的完整画面；MP4 编码不可用时回退到 AVI |
| `detections.jsonl` | 每个新检测帧的编号、时间、帧龄、检测框和置信度；XYZ 仅在与深度同帧时附带 |
| `depth_measurements.jsonl` | 每次完成的深度帧、源时间、计算耗时，以及该帧检测框对应的 XYZ |
| `frame_metrics.csv` | 采集 FPS、实际处理 FPS、帧龄及各阶段耗时 |
| `settings.json`、`camera.json` | 本次参数、计算设备与实际相机模式 |
| `summary.json`、`recording.json` | 处理与录像统计 |
| `calibration_snapshot.json` | 本次使用的标定快照 |

视频容器的 30 FPS 包含保持上一帧的编码，**不代表每秒有 30 个新识别结果**。

## 相机、标定与 XYZ

当前版本仅支持这套 USB 左右拼接彩色双目，使用 2026-10-13 的水下标定。RealSense/红外投影与旧棋盘格采样代码已移除。

- 采集：DirectShow / MJPG，目标 30 FPS，实际模式记录在 `camera.json`。
- 标定：`config/calibration.json`（保留完整质量与采样信息；已核对与原 YAML 几何一致）。
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
深度补洞只影响显示，XYZ 使用严格有效匹配点。
默认 `--depth-fps max`，深度异步不限频：前一任务完成后，在下一次检测迭代取最新双目帧继续计算。
最多保持一个未完成深度任务，检测不等待它完成；处理不过来时跳过旧帧，不积压任务。
`--depth-fps 5` 或 `10` 保留异步限频；`--depth-fps 0` 仍表示每帧等待同帧深度，不能当作异步不限频。
右目预处理也仅在需要更新深度时执行。不限频提高测距时效性，不改变单帧匹配算法或标定精度。
日志的 `depth_frame_id` / `depth_age_ms` 标明最新深度来源，`depth_fresh` 只表示深度帧是否新鲜，
具体目标的测距是否可用仍需检查 XYZ 是否为 null 以及 status；标定精度尚未验证。

2026-10-21，本机一次 CPU 实机运行（4189 帧、约 286 秒，录制开启）的基线：

| 指标 | 结果 |
|---|---:|
| 实际采集到达速度 | 平均约 17.9 FPS |
| 整段平均处理速度 | 约 14.6 FPS |
| 双目处理 | 平均约 52 ms |
| YOLO 推理 | 平均约 21 ms |
| 图像增强 | 平均约 9 ms |

本次重构后，相同相机设置下进行的两组 CPU 实机对照（每组 150 帧、预览开启、录制关闭）：

| 指标 | 同帧同步深度（`--depth-fps 0`） | 当时默认分频（`--depth-fps 5`） |
|---|---:|---:|
| 稳定新检测帧率 | 约 16.0 FPS | 约 18.0 FPS |
| 单帧处理耗时（排除前 10 帧） | 61.7 ms | 31.3 ms |
| 主机收帧到检测结果的平均帧龄 | 87.0 ms | 28.4 ms |
| 深度更新 | 约 15.9 Hz | 约 4.5 Hz |
| 跳过的相机帧 | 21 | 1 |

纯采集 150 帧测得约 17.94 FPS，驱动仍报告 30 FPS；当前实际新帧率受采集链路限制。
本机这次 Media Foundation（`--backend msmf`）初始化超时，默认保留已测通的 DirectShow。
上面的历史基线开启了录像，不能把两次不同条件的结果当作严格性能对照。
运行统计采用累计量和固定长度窗口，不随运行时间保留所有历史帧；完整逐帧记录写入 CSV/JSONL。

同日进一步排查，保持原 USB 扩展坞连接：

| 采集对照 | 实测 |
|---|---|
| 2560×720 双目，缩短手动曝光 | 仍约 18 FPS |
| 1280×480 左右画面 | 仍约 18 FPS，视野与原标定不同 |
| 1280×720 单视角画面 | 约 30 FPS；不能直接用于双目 XYZ |
| MSMF，关闭硬件转换后 | 返回约 30 FPS，但 300 帧中有 120 帧与上一帧完全相同，画面更新仍约 18 次/秒 |
| 恢复原曝光与 50 Hz 防闪烁设置后 | 2560×720 双目复测约 17.97 FPS |

USB 查询确认相机为 USB 2.0 High-Speed 设备；目前没有证据证明扩展坞带宽耗尽。
这轮未恢复完整双目真实 30 FPS，默认后端与标定保持原配置。
改变后端时，`--capture-only` 的返回帧率还需结合重复图像核查，不能仅凭数字判定新画面增加。
还需核对厂家对双目同时输出的规格，以及完成主机直连对照；不能据此断言硬件永久上限是 18 FPS。
详细记录见本机 `coral_runs/capture_diagnosis_20261021/REPORT.md`（运行记录不纳入 Git）。

上述处理有并行，耗时不能直接相加。帧龄是主机收帧到检测结束的统计，不包含相机内部缓冲、后续显示和通信，不是完整的相机到控制器延迟。
这些数据用于比较后续改动，不保证不同场景或电脑上的速度；当前没有新增 GPU 加速或跟踪控制接口。

2026-10-22，整理测距画面后，CPU、预览开启、录制关闭，各运行 150 帧的实机对照：

| 深度设置 | 最后窗口检测 FPS | 深度更新 FPS | 平均检测帧龄 / P95（排除前 10 帧） |
|---|---:|---:|---:|
| `5` | 17.94 | 4.48 | 36.7 / 47.0 ms |
| `10` | 17.90 | 8.60 | 33.5 / 47.0 ms |
| `max`（新默认） | 17.88 | 9.63 | 44.4 / 63.0 ms |

这是三次短时运行，不是长期性能保证。不限频增加了 CPU 竞争，检测帧龄有上升；后续 yaw 闭环可按延迟预算选择限频值。
记录见本机 `coral_runs/depth_rate_comparison/`。本日设备枚举中 USB 双目为 **相机 0**（相机 1 为虚拟摄像头），可用：

```powershell
uv run --locked python run.py --camera 0 --device cpu
```

相机编号可能随设备连接顺序改变；收到非 2560×720 画面时应检查设备选择，不要强行缩放后套用标定。

## 代码与依赖

| 文件 | 职责 |
|---|---|
| `run.py` | 实时/离线统一入口、参数和处理主流程 |
| `camera.py` | USB 采集进程、原子最新双目帧与实际帧率统计 |
| `stereo.py` | JSON 标定、SGBM/WLS、同帧 XYZ、公共补洞与 CPU 深度显示 |
| `gpu_depth.py` | NVIDIA CUDA 重建和内核 |
| `image_enhancement.py` | 图像增强 |
| `output.py` | 独立检测记录、画面绘制与录像 |
| `preview_window.py` | Windows 预览窗口比例约束与等比例显示 |
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

# 检查重构前后的标定、深度和 XYZ；需保留 Git 基线历史
uv run --locked python test_pipeline.py

# 检查四边/四角缩放、最小尺寸、最大化和关闭；不会打开相机
uv run --locked python test_preview_window.py

# 离线运行已有的原始左右图片，测试计算与编码，不会打开相机
uv run --locked python run.py --offline "D:\datasets\stereo" --frames 30 --device cpu --record-offline
```

纯采集测速输出到 `coral_runs/capture_日期时间/`，包含 `camera.json`、`capture_metrics.csv` 和 `summary.json`。
`--capture-fps` 设置请求采集频率（默认 30）；实际频率以测量为准。
`--exposure` 可按设备驱动单位指定曝光，默认不改当前曝光设置。不要把更短曝光带来的暗画面当作无代价提速。

`test_pipeline.py` 还会故意阻塞深度任务，检查检测仍能完成、队列不增长、历史 XYZ 不会混入当前检测框。
尚未新增 AMD GPU 后端、目标选择/锁定或 Teensy 控制接口。

离线目录需包含 `original/*_left.png` 和对应的 `*_right.png`，每张图片为 `1280×720`。
需要同时验证预览时追加 `--show-offline`。
