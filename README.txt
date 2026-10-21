水下珊瑚识别与 XYZ 测距 — uv 源码交付版
2026-10-21；保留 10 月 20 日已测试的画面增强和 GPU 提速代码。

一、安装与适用：Windows 64 位；普通 USB 彩色左右拼接双目相机，每目 1280×720。
电脑已安装 uv。本项目需要 uv 0.9.8 或更新版本，可先用 uv --version 检查。
若 uv 太旧，请通过原安装方式更新。本包完整运行环境实测使用 uv 0.13.4。
uv 会自动获取 Python 3.9.23，无需 Anaconda。

在本项目目录打开终端，首次安装执行：
    uv sync --locked

接好相机后启动：
    uv run --locked python run.py

点击相机画面使窗口获得键盘焦点：
    q / Q：开始录制
    w / W：保存视频并退出

录制视频：coral_runs/日期时间/visualization.mp4
同目录保存检测与 XYZ 数据、采集/处理指标、运行设置和标定快照。
默认相机索引为 1；如设备索引变化，指定实际双目设备编号，例如：
    uv run --locked python run.py --camera 2

CPU 运行：
    uv run --locked python run.py --device cpu

关闭画面增强：
    uv run --locked python run.py --input-enhancement off

二、首次安装说明
本包只含源码、模型、标定和依赖锁文件，不含 Python、.venv、依赖缓存或内置 uv。
首次 uv sync 需要联网：从 PyPI 和 PyTorch 官方 CUDA 11.6 索引下载锁定依赖，
其中 PyTorch 下载约 2.3 GiB。建议至少预留 12 GB 磁盘空间用于环境及安装缓存。
uv 会自动创建 .venv，并在用户目录管理 Python 和缓存。
已有 NVIDIA 显卡及兼容驱动时自动使用 GPU，否则使用 CPU；CPU 帧率会较低。
CUDA 运行库由 PyTorch 包提供，无需单独安装 CUDA Toolkit。
Windows VC++ 运行库和 NVIDIA 驱动属于系统组件。
首次安装完成后，保持依赖版本不变；不要随意执行 uv lock --upgrade。

三、确认相机与标定
这是普通 USB 彩色双目 1280×720 版本，不是 D435i 的 640×480 标定。
采集方式：DirectShow / MJPG；总帧 2560×720 = 左右各 1280×720；目标 30 FPS。
图像左右顺序 LEFT|RIGHT，不交换左右目。
标定文件：config/stereo_opencv.yaml 和 config/calibration.json。
两者与这套相机 2026-10-13 的原始水下标定一致。
基线：59.55734255 mm。
原始左目 K1：fx=1406.8097219，fy=1409.6292648，cx=634.4973071，cy=490.6416794。
原始右目 K2：fx=1411.7321975，fy=1413.6009901，cx=686.5858993，cy=483.0347250。
校正后 P1：fx=fy=1688.2932880，cx=797.5483360，cy=471.8612728。
XYZ 使用校正后的图像坐标与 P1，不能把原始 K1 直接套到校正图上。
原标定 quality_passed=false 状态保留；物理测距精度仍需用已知距离实测确认。

四、保留的显示与性能配置
左侧：增强彩色画面、检测框、类别、置信度、X/Y/Z（cm），保留已确认的加粗标签。
右侧：纯深度图和色标。
默认 mild 增强、native CUDA、FP16、OpenCV 2 线程、YOLO imgsz416、置信度至少 0.60。
双目匹配工作尺寸 320×180，保持 1280×720 的输出视野；标定分辨率没有改变。
增强只用于检测输入和左侧显示，原始左右目仍用于双目匹配与几何测量。
深度补洞只影响显示，XYZ 使用严格有效匹配点。
坐标原点为校正后的左相机光心，X 向右、Y 向下、Z 向前；内部单位 m，标签单位 cm。
X=(u-cx)*Z/fx，Y=(v-cy)*Z/fy；u/v 为校正图像像素坐标。
目标 XYZ 为检测框中心区域有效三维点的稳健中位数；Z 是前向深度。

原机器 GTX 1650 Ti 实测：相机采集约 30 FPS，完整实时识别测深约 13 FPS。
本次 uv 环境离线测得约 16.3 FPS，与昨日同组图片约 16.4 FPS 基本一致。
视频容器的 30 FPS 包含保持上一帧的编码，不能当作新识别结果帧率。
不同电脑的实际处理帧率取决于硬件；此处保留相同代码与参数，不保证相同速度。

五、文件与依赖
11 个根目录 .py 文件：全部运行模块。
models/best.pt：珊瑚模型；config/：对应水下内外参。
pyproject.toml：项目与依赖配置；uv.lock：锁定版本、来源和哈希。
.python-version：Python 3.9.23；MANIFEST.sha256：交付文件校验。
测试素材、旧录像、验证输出、历史备份、内置工具和一键脚本已从传输包删除。

核心依赖保持已测试版本：torch 1.13.1+cu116、torchvision 0.14.1+cu116、
ultralytics 8.3.199、numpy 1.26.4、opencv-contrib-python 4.13.0.92。
uv 显式排除 opencv-python，由 contrib 唯一提供 cv2 和 ximgproc，避免重复覆盖。
OpenCV 发布元数据声明 numpy>=2；为兼容旧 PyTorch，使用实测的 NumPy 1.26.4 覆盖。
打包前已通过 45 项回归测试及 GPU/CPU 离线全流程；本次清理未更改运行逻辑、模型或标定。
uv 依赖排除功能最低版本说明：https://github.com/astral-sh/uv/releases/tag/0.9.8
