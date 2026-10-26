# 水下珊瑚识别与 XYZ 测距

普通 USB 左右拼接双目相机的实时珊瑚检测、图像增强和水下标定测距程序。使用 uv 管理 Python 和依赖。

后续开发见 [珊瑚 yaw 视觉闭环计划](PLAN.md)：方案 A 为视觉偏差 → 目标航向变化速度 → Teensy 更新目标 yaw → 原有双环。支持本地 USB 或 Jetson RTP 网络双目输入；Controller 仍通过独立 USB 连接 Teensy。

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

## 工控机网络视频

相机接 Jetson，网线连接 PC；Teensy 仍单独通过 USB 接 PC。只运行 Jetson 的 `video_udp.py`，本方案不需要控制 Gateway 主程序。视觉、Rov_gateway、Controller 的开发分支为 `codex/network-video`；固件继续使用已验证的 `codex/local-visual-yaw`，无需重新烧录。Controller 修正 Windows 粗时钟造成的发送降频，协议及 PID 不变。

先在 Jetson 终端启动（PC 有线地址 `192.168.10.1`，Jetson `192.168.10.2`）：

```bash
cd /home/nvidia/Code/python/rov_gateway
/home/linuxbrew/.linuxbrew/bin/uv run python video_udp.py --preset stereo-720 --input-format mjpeg --encoder nvv4l2h264enc --host 192.168.10.1 --port 5600
```

再在 PC 本项目目录运行：

```powershell
uv run --locked python run.py --stream-host 192.168.10.2 --device cpu --confidence 0.5

# 单独检查网络收帧，不加载识别/深度或发送控制
uv run --locked python run.py --capture-only --stream-host 192.168.10.2 --frames 150
```

需要 PATH 中的 `ffmpeg`（本机已验证 FFmpeg 8.0），也可用 `--ffmpeg C:\path\ffmpeg.exe` 指定。`uv sync` 不安装此系统程序。无需重编译 OpenCV 或安装 Python GStreamer 绑定。Windows 防火墙须允许来自 Jetson 的 FFmpeg UDP 接收；RTP 为5600，RTCP占用5601，端口可用 `--stream-port` 改，但发送端要同步修改。相同端口只能给一个接收程序使用；先关闭 GStreamer/FFplay 测试播放器。

网络参数替代 `--camera`，不是把相机编号改成 URL。必须发送 `2560×720 LEFT|RIGHT`、H.264、RTP payload 96；尺寸不匹配直接报错，不自动拉伸。`--capture-fps`、`--backend` 是本地 USB 参数，网络帧率在 Jetson 发送端配置。识别416、深度320、异步不限频和现有标定保持不变；H.264 对测距精度的影响仍需已知距离对照。

Controller 的启动方式不变，在它的目录运行：

```powershell
uv run python ds5_teensy_demo.py --teensy-port COM3 --vision --vofa
```

先确认自稳模式，再点击珊瑚；右键/C取消。网络接收仍只保留最新完整左右帧，未增加控制协议或另一套 PID。顶部 `RTP` 表示网络输入，画面过期显示 `RTP STALE - CONTROL INVALID`。

时间戳规则：FFmpeg 解码后的每一帧与其 PTS 一一配对，按本次会话最早到达基线映射到 PC 单调时钟。`frame_metrics.csv` 新增 `received_monotonic_s`、`stream_pts_s`、`delay_growth_ms`；`capture_monotonic_s` 在网络模式下是这个映射时间，**不是曝光时间**。已积压的帧不会因为刚解码而重新取得完整有效期，但启动时已有的固定延迟不可由此测得。完整相机→Teensy延迟仍须端到端测量，不能把日志的帧龄当作该值。

帧过期沿用200ms撤销机制；画面恢复后须重新点击。连续5秒没有新解码帧，或源PTS回退/突跳、尺寸改变、解码进程异常时退出并释放接收端；重新启动程序后再选目标。5秒是接收程序退出期限，**不是继续控制5秒**。首版不做后台自动重连。

软件检查（不连接串口）：

```powershell
uv run --locked python test_network_video.py --integration
```

该检查覆盖真实本机 RTP/FFmpeg 解码、慢消费者跳帧、短暂断流后不自动恢复锁定、断流超时、端口释放及非法视频时间戳。

2026-10-26 实测：按用户要求在约5分钟结束连续观察，处理4633帧，平均16.12 FPS；接收延迟增长P95约28.9ms，映射帧龄P95约66.7ms（均不是曝光到执行延迟）。后续重启推流的台架测试中实际接收约30 FPS、识别约26 FPS，不能据此保证所有曝光/场景都到30 FPS。最终真实Jetson→PC识别→Controller→COM3 Teensy无动力测试约50秒，收到1866份活动回执；三次脚本选定、取消、0.6秒推流暂停、恢复后重新选择及彻底停止推流全部验证，除主动故障外没有额外超时。Controller发送实测48.08Hz，八路PWM始终1500，结束确认MANUAL/LIMIT0。记录在本机 `coral_runs/network_bench/verified_result.json`；这是脚本调用选择入口的无动力检查，不替代物理手柄/鼠标、水下运动、已知距离或端到端时延验收。

## 窗口与录制

点击预览窗口使其获得键盘焦点：

| 操作 | 效果 |
|---|---|
| 左键点击左目检测框 | 选择同一株珊瑚进行关联，重叠框有歧义时拒绝选择 |
| 右键或 `C` | 取消选择 |
| `Q` | 开始录制 |
| `W` 或 `Esc` | 保存录像并退出 |
| 拖动窗口四边或四角 | 宽高按画布比例联动缩放 |
| 最大化或贴靠窗口 | 画面等比例显示，空余区域留边 |

默认只显示左目增强彩色画面、检测框、类别和置信度，右目继续在后台参与异步测距。
识别尺寸保持 `--imgsz 416`，深度匹配保持 `--stereo-width 320`；相机每目仍为 `1280×720`。
顶部只显示模式、控制状态与检测 FPS；底部三行显示当前锁定目标 XYZ（cm）、目标航向变化速度的 SENT/TEENSY 值、目标/实际 yaw 与领先角上限。XYZ 为同一锁定目标最近一次有效测距，并标注源帧年龄；不同锁定、无有效点、漏检或过期时显示 `--`，不使用深度显示补洞值。

黄色 SELECTED 表示视觉选中，青色 CONTROL 表示收到启用回执；短暂漏检显示橙色虚线。SENT 是最近一次实际组包发送的值，TEENSY 是新鲜回执值；没有确认显示 `--`，已确认关闭显示 0。标定未验证时在测距旁提示，X向右/Y向下/Z向前。详细帧编号、偏差和耗时留在日志。

需要查看深度伪彩图和 cm 色标时，启动时增加调试参数：

```powershell
uv run --locked python run.py --camera 0 --device cpu --show-depth
```

左目图像始终为 `1280×720`（`16:9`），含上下信息栏的默认画布为 `1280×928`；
启用深度调试面板后为 `2688×928`。窗口始终按所选完整画布的比例缩放，图像不会拉伸。
缩放限制考虑标题栏和边框；预览缩放不改变标定、计算或录像尺寸。
隐藏深度面板只改变显示，后台深度算法与异步调度不变。

每次运行输出到 `coral_runs/日期时间/`。诊断 PNG/NPY 只保留固定数量快照，退出控制后再编码保存，避免写图阻塞新观测；运行中仍持续写 CSV/JSONL：

| 文件 | 内容 |
|---|---|
| `visualization.mp4` | 按 Q 后保存的完整画面；MP4 编码不可用时回退到 AVI |
| `detections.jsonl` | 每个新检测帧的编号、时间、帧龄、检测框和置信度；XYZ 仅在与深度同帧时附带 |
| `control_feedback.jsonl` | Controller 回执、停止原因、发送/接受值和实际航向，便于定位偶发退出 |
| `vision_observations.jsonl` | 每帧及选择/取消/过期/退出事件的视觉观测与候选转向预览 |
| `preview_locked.png` | 本次首次有效锁定时的原始完整预览 |
| `depth_measurements.jsonl` | 每次完成的深度帧、源时间、计算耗时，以及该帧检测框对应的 XYZ |
| `frame_metrics.csv` | 采集 FPS、实际处理 FPS、帧龄及各阶段耗时 |
| `settings.json`、`camera.json` | 本次参数、计算设备与实际相机模式 |
| `summary.json`、`recording.json` | 处理与录像统计 |
| `calibration_snapshot.json` | 本次使用的标定快照 |

视频容器的 30 FPS 包含保持上一帧的编码，**不代表每秒有 30 个新识别结果**。

## 目标锁定与 yaw 请求预览（第二步）

左键点击检测框选择目标，右键或 C 取消。仅运行视觉时只观察；Controller 加 `--vision` 后，已经确认自稳时的新点击才请求启用。顶部明确区分等待确认、跟踪、居中保持、超时、人工接管和取消。中央三条虚线表示参考点与视觉默认死区。

首版 yaw 不依赖 XYZ。深度任务保存其源帧当时的锁定 ID 和目标框，完成后只将同帧匹配所得的 XYZ 提供给相同锁定；切换目标不会继承旧距离。最多保留至 `--depth-max-age`（默认0.75秒），显示年龄从主机收到源帧计时，不是曝光到执行延迟。

```text
error_x = (目标中心 u - reference_u) / (图像宽度 / 2)
alpha = 1 - exp(-源帧时间间隔 / 0.050秒)
smoothed_error_x = 上次平滑偏差 + alpha × (error_x - 上次平滑偏差)
preview = yaw_sign × sign(smoothed_error_x) × min(yaw_max_slew, yaw_gain × max(0, abs(smoothed_error_x) - yaw_deadband))
```

默认参考点为图像宽度的 50%，不是标定光心；例如 u=800、宽度1280，对应 error_x=+0.25。
日志中的归一化偏差相对半幅宽度，不能当作角度。
首次选择或稳定在此位置时，此例预览为 **-2.10°/s**。平滑时间常数默认50ms，可设为0关闭；首次选择和等待后恢复时重置平滑，不沿用旧位置。控制和日志中的候选预览使用平滑偏差，日志同时保留原始偏差。负号按当前 yaw 约定用于向右转向的候选映射，
须结合相机安装和艇体 yaw 正负方向完成台架确认；这些数值尚未用于水下闭环整定。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--confidence` | 0.60 | 检测框显示、目标初选、等待后恢复及测距目标筛选门限；范围 `0 < 值 <= 1` |
| `--yaw-reference` | 0.5 | 期望水平位置 / 全图宽度 |
| `--yaw-gain` | 10 | 每单位归一化偏差对应的目标航向变化速度，°/s |
| `--yaw-deadband` | 0.04 | 半幅宽度的4%，等价于中心左右各全图宽度2% |
| `--yaw-max-slew` | 5 | 目标航向变化速度幅值上限，°/s |
| `--yaw-sign` | -1 | 候选方向系数，可设为 +1，必须台架核对 |
| `--vision-max-age` | 0.2 | 最大主机帧龄及连续关联的最大源帧间隔，秒 |
| `--track-min-iou` | 0.2 | 后续框与前一选定框的最小交并比 |
| `--track-max-motion` | 0.5 | 最大中心位移 / 前一框对角线长度 |
| `--track-confidence` | min(0.35, confidence) | 连续跟踪门限；可单独设置，但不能高于 `--confidence` |
| `--track-hold` | 0.4 | 从最后一次接受的检测起保留目标身份的秒数；等待期间请求无效 |
| `--track-smooth-ms` | 50 | 黄色框与候选预览的指数平滑时间常数，毫秒；0关闭 |
| `--vision-port` | 14560 | 本机 UDP 目标端口；0关闭发送；离线模式始终不发送 |

候选框须类别一致，且同时满足重叠与位移门限；只有一个候选才继续锁定。
默认连续跟踪允许置信度降至0.35，新目标选择和测距目标筛选使用 `--confidence`（默认0.60，可降低）。无候选时进入 WAITING，保留原锁定 ID 和最后位置；默认最多保留至最后一次接受检测后的0.4秒。期限内连续两帧出现满足关联条件、达到 `--confidence` 的唯一候选，才恢复同一 ID。弱候选或一次重现不能延长期限。
超过保留期限、多个候选造成歧义、源帧过期/断流、逆序帧或人工取消，均结束锁定，需重新点击。0.4秒只容忍检测漏检，不放宽0.2秒的图像时效门限。
该方法没有外观重识别，交叉遮挡可能造成身份混淆，需用实际录像验证；它优先拒绝可见歧义。
等待期间橙色虚线框仅表示历史位置；不会以历史位置或预测值维持转向。平滑会引入少量延迟，默认参数尚需水下录像验证。
无效预览显示 `-- (INVALID)`，输出数值为 null；有效且居中时才输出 0°/s。

最终方案 A 中，**Controller** 校验观测并重新计算转向请求，Teensy 执行
`yaw_target += yaw_target_slew_dps × dt`，再走现有角度外环和角速度内环。
此处预览不等同于艇体角速度目标，也不提供绝对 yaw；居中滞回、目标角收拢、领先量限制和手柄仲裁属于后续控制接入。

### 本机观测接口

实时模式每个检测帧先于深度等待、绘图和录像发送一个 UTF-8 JSON UDP 数据报到 `127.0.0.1:14560`，
不绑定串口，不经 VOFA。选择、取消、已锁定目标过期及正常退出也发布状态事件。
UDP 非阻塞、无重试队列；失败在界面标为 UDP ERROR，计数写入 summary。
所有观测另写入 `vision_observations.jsonl`；强制杀进程无法保证发出最终无效包，接收端必须独立超时。

| 字段 | 含义 |
|---|---|
| `schema` | 固定 `coral.observation.v1` |
| `session`、`observation_seq` | 每次启动的新会话和发送序号；事件序号递增不代表有新图像 |
| `capture_seq`、`lock_id` | 源图像编号、人工选择生成的锁定编号 |
| `selection_monotonic_s` | 人工选择时间，短暂恢复不更新；取消/真丢失后为 null，用于拒绝启动、接管或模式切换之前的旧点击 |
| `valid`、`status` | 有效标记及 idle/tracking/waiting/lost/ambiguous/stale/cancelled/no_target/stopped |
| `capture_monotonic_s`、`result_monotonic_s` | PC 收到源帧和生成这条观测的单调时钟时间（秒） |
| `source_age_ms`、`valid_for_ms` | 发送时的源帧年龄及从源帧时间起算的总有效期 |
| `image_size`、`coordinate_frame` | 每目尺寸及固定 `rectified_left` 坐标系 |
| `reference_u`、`principal_u` | 期望横坐标与校正后标定光心，分别记录 |
| `target`、`center_uv`、`error_x` | 选定框/类别/置信度、中心、归一化水平偏差；无效时为 null |
| `smoothed_center_uv`、`smoothed_error_x` | 平滑中心和偏差；预览使用此偏差，无效时为 null；原始字段保持原语义 |
| `track_memory` | 仅供显示的框、最后接受检测时间、年龄、保留剩余时间及恢复计数；`display_only=true`，禁止用于控制 |
| `preview` | 候选 `yaw_target_slew_dps` 及 gain/deadband/limit_dps/sign/smooth_ms，`sent_to_teensy=false` |

配套 Controller 检查 schema、数值范围、会话、锁定变化、序号和源帧年龄；
WAITING 即使源图像新鲜也按 `valid=false` 停止视觉修正，所有偏差及候选请求均为 null。Controller 使用 `smoothed_error_x` 计算请求，避免再次叠加同类平滑；`track_memory` 不能作为有效观测。
接收时剩余有效期按原源帧时间扣减，不能从收包时重新计时，也不能用 observation_seq 代替 capture_seq 判断新帧。
重复发送或点击同一源帧不能延长其有效期。只有同 PC 单调时钟可直接比较；时间戳不代表曝光或视频端到端延迟。
`preview` 用于观察与日志，不能作为不经仲裁直接发给执行器的指令。离线日志和实时日志由 settings.json 的 mode 区分。

```powershell
# 保持原来的异步深度与默认分辨率
uv run --locked python run.py --camera 0 --device cpu

# 显示、初选及恢复门限改为50%；续跟默认仍为35%
uv run --locked python run.py --camera 0 --device cpu --confidence 0.5

# 同时调整两个门限：初选/恢复45%，续跟30%
uv run --locked python run.py --camera 0 --device cpu --confidence 0.45 --track-confidence 0.3

# 仅验证识别和 yaw 观测，明确关闭深度计算
uv run --locked python run.py --camera 0 --device cpu --no-depth

# 不发送 UDP，仍显示并记录观测
uv run --locked python run.py --camera 0 --device cpu --vision-port 0

uv run --locked python test_vision.py
```

置信度参数用0到1的小数，例如50%写作 `0.5`，不能写 `50`。只设置 `--confidence 0.3` 时，续跟门限自动降至0.3；显式设置的续跟门限若高于初选门限会报错。不传参数时保持原来的0.60/0.35。
降低门限可能减少弱检测导致的漏检、加快等待后的恢复，但不保证身份更稳定：误检和多个候选造成的歧义也可能增加。可先用0.50与默认0.60对照；若问题是框跳动、遮挡或类别变化，单调门限无法解决。检测置信度与双目匹配的 `--wls-confidence` 无关，深度有效性检查保持不变。

## 本地视觉 yaw 辅助

Teensy 使用 `codex/local-visual-yaw` 配套固件；当前视觉和Controller在 `codex/network-video` 上兼容本地USB输入。两个终端分别启动：

```powershell
# Controller 目录：COM3 为本次实测端口，按实际枚举修改；--vofa 可选
uv run python ds5_teensy_demo.py --mode serial --teensy-port COM3 --vision --vofa

# 本视觉项目目录
uv run --locked python run.py --camera 0 --device cpu --confidence 0.5
```

Controller 仍读取 DS5，默认50Hz发包。按 R1 进入自稳、释放旋转键后，再点击珊瑚。只有新鲜遥测确认已经自稳时的新点击才允许启用；提前锁定后切模式、重启或重连均须重新点击。右键/C取消，方框或圆圈键立即接管 yaw；松开不自动恢复。前后、横移和升降仍由驾驶员操作。短暂 WAITING 停止修正并保持航向，期限内确认同一目标可恢复；真正丢失须重新点击。

`CONTROL ACTIVE` 表示固件已接受修正，`CENTERED` 表示居中保持，`WAITING` 表示暂停修正，`MANUAL_OVERRIDE/CLICK_AGAIN/WAIT_LINK` 表示接管、需重新选择或无新鲜回执。确认值可能稍晚于当前检测，不等同于艇体已经转到位。XYZ 不参与 yaw。

实际控制参数由 Controller 的 `--vision-gain 10 --vision-limit 5 --vision-sign -1 --vision-deadband 0.04 --vision-exit-deadband 0.06` 决定；本程序 `--yaw-*` 只改变候选预览。进入/退出居中阈值相对半幅宽度，滞回避免重复收拢航向。固件另有限幅5°/s、目标领先15°及200ms独立时效。Controller预留20ms串口传输预算；真实传输超出预算仍需测量，主机帧龄不是曝光到执行延迟。

观测发往 `127.0.0.1:14560`，Controller 从同一socket回复 `coral.control.v1` 到视觉源端口。回执包含会话/锁定ID、确认状态、请求值、已接受值与截止时间；过期不显示为生效。没有新增线程或历史观测队列。

跨项目软件检查，无需连接相机、Teensy或DS5：

```powershell
# 在 ROV_Teensy41 目录生成测试用主机程序
python tests/run_host_checks.py --export-vision-sim ../uv20261021/coral_runs/control_link/firmware_sim.exe
# 在本视觉项目目录
uv run --locked python test_control_link.py --firmware-sim coral_runs/control_link/firmware_sim.exe
```

2026-10-23 已烧录配套固件并完成真实Teensy/IMU无动力检查：启用、等待恢复、居中、取消、手动接管、断帧超时和模式重入均通过。随后真实相机240帧与Controller/USB联调，检测约17.90 FPS、深度9.21 Hz；实际识别目标由测试脚本调用同一选择入口，收到617份有效控制回执，八路PWM全为1500。记录位于本机 `coral_runs/control_link/`。

后续修复：逐包采样接收时间、按观测序号和锁定编号处理同批次/同时间戳的取消与新点击，保留退出原因；去除实时循环中的同步诊断图片编码。新版固件领先角15°，新增目标/实际航向回执。`revision_2/` 台架记录中连续12次取消重选全部启用，实际领先角达到15°；VOFA收到手动调参OK、自稳调参MODE_REJECTED和四通道曲线，八路PWM均1500。真实相机持续复测600帧，收到1337份活动回执，未发生超时撤销；这是无动力、脚本调用选择入口的验证，仍需人工与水下验收。

测试结束保留 **MANUAL、推力上限0**。没有进行推进器带动力转动或水下整定；恢复动力前按原流程检查推力上限与方向。

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

默认使用 mild 增强、YOLO `imgsz=416`、新目标/测距置信度门限 `0.60`（可调）、OpenCV 2 线程；推理按续跟门限保留候选，默认低至 `0.35`。
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
这些数据用于比较后续改动，不保证不同场景或电脑上的速度；这些历史数据对应当时版本；没有新增 GPU 加速。

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
| `camera.py` | USB/网络采集进程、原子最新双目帧与实际到达帧率统计 |
| `network_video.py` | FFmpeg RTP/H.264 接收、PTS 配对与延迟增长检查 |
| `stereo.py` | JSON 标定、SGBM/WLS、同帧 XYZ、公共补洞与 CPU 深度显示 |
| `gpu_depth.py` | NVIDIA CUDA 重建和内核 |
| `image_enhancement.py` | 图像增强 |
| `vision.py` | 单目标关联、水平偏差与请求预览、本机 UDP 和观测日志 |
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
尚未新增 AMD GPU 后端；Teensy 控制由配套 Controller 独占串口完成。目标选择、观测输出与请求预览检查见 `test_vision.py`。

离线目录需包含 `original/*_left.png` 和对应的 `*_right.png`，每张图片为 `1280×720`。
需要同时验证预览时追加 `--show-offline`。
