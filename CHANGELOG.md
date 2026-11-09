# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.4.0] - 2026-11-09

从"图片 + 系统配音 + 硬编成片"升级为**全本地 AI 生产管线**：声音克隆替掉 Windows SAPI、
人物有骨骼绑定会动、能按参考视频生成相似视频、音效库与素材库、剧本生成模型、LobeHub 对话客户端。

### Added

- `theater/tts_clone.py`：F5-TTS 1.1.22 的声音克隆。`ENGINE=auto` 时有权重就走 clone、
  缺参考音或参考文字时自动回落 SAPI。含 `cache_key()`（把参考音频的绝对路径 + 大小一起进
  key，换参考音不会误命中旧缓存）、`_vocoder_local_path()`、`_patch_f5_audio_io()`
  （把 `torchaudio.load` 换成 soundfile，绕开 torchcodec 对共享版 ffmpeg 的依赖）、
  `status()`、`parse_voice_map()`。
- `theater/rigging.py`：解剖先验的 2D 伪骨骼。按 alpha 包围盒切 头/躯干/双臂/双腿，
  每部件记 bbox + 关节 pivot + z 序，落 `rig.json`；同一张图两次 build 签名一致。
- `theater/animator.py`：`ACTIONS`（idle/nod/wave/walk/talk）、`pose_at(action, t)` 纯函数、
  `render_frame()`、`render_clip()`（写 PNG 序列 + `clip_manifest.json`，含 `joint_tracks`）、
  `frame_diff()`。
- `theater/library.py`：素材/音效库。sha256 去重入库、`index.json` 索引、中英同义词检索
  （`search` / `pick_for_prompt`）、9 种现场合成音效（rain/whoosh/click/typing/door/thud/
  incoming/heartbeat/chime）、`mix_sfx()` amix 叠轨、`ensure_builtin_sfx()`、`stats()`。
- `theater/video_ref.py`：参考视频 → 相似视频。`analyze_reference()` / `plan_frames()`
  （超上限就等间隔抽帧再复制回原帧率，保住时长）/ `edge_iou()`（相似度量）/ `_restyle()`
  （SD1.5 img2img）/ `generate_similar_video()` / `similarity_report()`。
- `theater/frame_compose.py`：新增 `compose_actor_frame()` —— 用"某一时刻带动作的角色图"
  合成整帧 1920×1080 并叠字幕；站位不认识时按居中回落，不报错。
- `pipeline_full.py`：`STEPS` 扩到 `parse → assets → tts → sfx → anim → frames → video`；
  新增 `build_sfx()`（台词关键词 → 内置音效，命中不了再问库）、`build_anims()`
  （发言角色 `talk`、其余 `idle`/`nod`，时长以该镜头配音总时长为准）、`_sfx_name_for()`。
- `pipeline_full.py`：剧本模型双后端。`LLM_BACKEND=auto|torch|openvino`，torch 读
  `MODEL_ROOT/qwen3-4b`（bf16 + `apply_chat_template(enable_thinking=False)`），
  OpenVINO 按 `_ov_device()` NPU→GPU→CPU 逐级挑真实存在的设备。
- `server.py`：MCP 工具 8 → 13，新增 `theater_clone_voice`、`theater_animate_character`、
  `theater_generate_similar_video`、`theater_library_search`、`theater_make_sfx`；
  `theater_export_video` 增加 `animate` 开关并把 `sfx`/`anim` 步骤纳入默认链路。
- `mcp/lobe-mcp.json`：LobeHub 直接可用的 MCP 配置（13 个工具）。
- `README.md` 第 13 节：六项能力对照、全 D 盘安装、新增环境变量、本机实测、踩坑表、
  客户端接入、五条复验命令。
- `tests/test_library.py`、`tests/test_rigging_animator.py`、`tests/test_tts_and_video.py`、
  `tests/test_integration.py`（新增 28 条，总数 25 → 53）。
- `_tools/`：`mk_manifests.py`、`fix_deps.py`（缺什么补什么，带 google/triton/deepspeed 黑名单）、
  `cached_path_shim.py`、`make_ref_voice.py`、`smoke_clone.py`、`probe_*`、
  `pdl2.py`（分段下载逐段校验 + 整文件 sha256）、`e2e_demo.py`（2 镜头全链路自测）。

### Changed

- `theater/local_tts.py`：SAPI 发音人映射改名 `SAPI_VOICE_MAP`（`TTS_VOICE_MAP` 专供克隆引擎，
  值是参考音频文件名），两者不再撞车；`gen_all_tts()` 在 clone 可用时整体委托给 `tts_clone`。
- `theater/video_export.py`：`_build_clip()` 支持 PNG 序列（帧数不够自动循环铺满）与音效垫底；
  `export_final_video(..., anims=None, sfx_map=None)` 返回 `(path, duration)`。
- `theater/ai_draw.py`：`LIBRARY_FIRST`（默认开）。通用背景 / 每镜头背景 / 角色立绘
  都改成"先查库、命中就复制、否则自绘并把产物入库"，跑得越久复用率越高。
- `check_env.py`：新增克隆链依赖、F5 权重、vocos、SD1.5、Qwen3-4B（torch/OV 双查）、
  配音引擎、骨骼动作、素材库、参考视频重绘等检查项。
- `requirements.txt`：补声音克隆整条依赖链与三条硬约束（`--no-deps`、`datasets==3.6.0`、
  `cached_path` 用 shim），以及 SD1.5 和 Qwen3-4B 的落地说明。

### Fixed

- `theater/library.py::_synth_samples("typing")`：末尾 burst 切片短于数组长度，
  会 `ValueError: operands could not be broadcast together`；改为按 `n - ln` 取起点。
- `theater/video_export.py::_build_clip()`：concat demuxer 不认 `-framerate`
  （那是 image2 的选项），改 `-r <fps>`；`-shortest` 会把镜头截到较短音轨长度
  （本机实测 6.0 s 被截成 4.7 s），改为 `-af apad` + `-t <时长>`。
- `server.py`：mcp 2.x 把 `FastMCP` 改名成 `MCPServer`（`mcp.server.mcpserver`），
  改为先试 1.x 再 fallback，两代都能起。
- 补 `D:\ai_models\sd15-base\feature_extractor\preprocessor_config.json`
  （只下了 unet/vae/text_encoder 时 diffusers 会报 `Can't load image processor`）。

### Verified

- `python -m unittest discover -s tests` → **53 tests OK**。
- 声音克隆：参考音 `cache/voices/zhou_ref.wav`（822,014 B，SAPI 合成）+ 同名 `.txt`；
  单句 22.3 s（含首次加载），产物 24 kHz / mono / pcm_s16le / 5.696 s；
  批量 2 句 12.4 s（模型常驻），`engine=clone`。
- 骨骼：512×512 立绘 → 6 部件 7 关节，签名 `4a096c50a50e`；`wave` 2 s / 24 fps 渲 48 帧。
- 全链路（`_tools/e2e_demo.py`，2 镜头）：克隆配音 + 音效（click/typing）+ 144 帧动画
  → `cache/demo_animated.mp4`：6.02 s / 1920×1080 / 30 fps / 180 帧 / 1,128,400 B，
  h264_nvenc 编码 5.1 s；抽 0.2 s 与 1.0 s 两帧 `mean_abs_diff=0.426`、0.88% 像素在动
  —— 人物确实在动，不是静帧。
- 剧本生成：Qwen3-4B bf16 在本机 15.9 GB 显存全量驻留，加载 147 s、生成 6 镜头 32 s。
- 参考视频 → 相似视频：拿 6.02 s 的成片当参考，抽 24 帧逐帧 img2img（strength 0.45 /
  8 步 / 先缩到 512 再放大回原尺寸），**24.2 s** 出 `cache/similar_demo.mp4`
  （6.0 s / 30 fps / 1920×1080 / 180 帧 / 保留原音轨）。`similarity_report()`：
  时长偏差 **0.0%**、fps 一致、分辨率一致、边缘 IoU 均值 **0.344**（strength 0.45
  本来就会改掉细节纹理，IoU 只用来判"构图没漂"，不是像素级复刻指标）。
- 素材库：9 个内置音效入库 0.7 MB，`library.stats()` 可查；检索「下雨 雨声」命中 `rain`。

## [1.3.0] - 2026-11-08

全 D 盘安装 + 首次真机跑通（Python 3.12 / CUDA torch / 带 NVENC 的 ffmpeg / FLUX 权重全部落在 D 盘）。

### Added

- `theater/__init__.py`：`ffmpeg_path()` / `ffprobe_path()` 自动查找，候选路径依次为
  `FFMPEG_PATH` → `项目根\tools\ffmpeg\bin` → `项目根\tools\ffmpeg` → `_tools\ffmpeg\bin`
  → `_tools\ffmpeg` → `D:\ffmpeg\bin` → `D:\ffmpeg` → `PATH`。不再要求用户手动配环境变量。
- `TORCH_MODEL_DIR`（默认 `$MODEL_ROOT/flux-schnell`）：torch 后端专用的 diffusers 目录，
  与 OpenVINO 的 `DRAW_MODEL_DIR` 分开，避免"INT4 IR 喂给 torch"这类格式错配。
- `README.md` 第 12 节「全 D 盘安装」：目录结构、五步安装脚本、模型下载命令、
  四条验证命令、本机实测数据、网络可用性对照表、16 GB 显存跑 FLUX 的 offload 策略。
- `requirements.txt` 补 cu128 与 D 盘安装提示。

### Changed

- `theater/ai_draw.py`：`_load_torch_pipe()` 现在按后端取模型目录；CUDA 下用 `bfloat16`
  （原 `float16`）；显存 < 24 GB 时**默认 sequential offload**（按本机实测，`model_cpu_offload`
  会让进程直接崩，见下）；目录缺 `model_index.json` 时给出"这不是 diffusers 目录"的明确报错。
- `theater/ai_draw.py`：抠图改为**显式 `new_session("u2net")`**。rembg 2.x 的 `remove()`
  默认模型已换成 birefnet（约 1 GB），不指定就会联网下载，既破坏离线也会写到默认目录；
  现在 `U2NET_HOME` 钉在 `MODEL_ROOT/rembg` 并只认本地 `u2net.onnx`，缺文件直接降级白底抠图。
- `theater/ai_draw.py`：`generate_scene_assets()` 返回的 `model_dir` 改为**实际生效的后端目录**
  （torch 报 `flux-schnell`，OpenVINO 才报 `flux-schnell-int4-ov`），不再写死 OpenVINO 路径。
- `check_env.py`：
  - **修 bug**：NVENC 真编测试原来用 `64x64` 黑场，NVENC 有最小帧尺寸限制，会报
    `Frame Dimension less than the minimum supported value`，把能用的编码器误判成不可用；
    改为 `320x240`。
  - 模型检查改为按当前后端选目录（torch 查 `model_index.json`，OpenVINO 查 `*.xml/*.bin`），
    另一套格式只作"备用模型"提示，不再因为没装 OpenVINO 模型就 FAIL。
- `.gitignore` 增加 `tools/`（ffmpeg 等本地大二进制不入库）。

### Verified

- `check_env.py`：FAIL 0 项（WARN 2 项为 videotoolbox 不适用、Qwen3 剧本模型未下）。
- `torch 2.9.1+cu128` + `torch.cuda.is_available() == True`，RTX 5060 Ti / 15.9 GB / bf16 支持；
  bf16 matmul / conv / SDPA 三个基础 kernel 在 GPU 上真跑通过。
- ffmpeg `N-128691`（2026-11-08，BtbN GPL 构建）：`h264_nvenc` / `hevc_nvenc` / `av1_nvenc` /
  `h264_amf` / `hevc_amf` / `libx264` 六种编码器**真编**通过。
- **FLUX 真出图**（`torch` / `cuda` / `bfloat16` / sequential offload）：1024×1024 一张 1504 s，
  512×512 一张约 21 分钟；`enable_model_cpu_offload()` 在本机 **推理第一步即 `0xC0000005`**，
  因此默认改走 sequential。
- **抠图真跑**：u2net 离线抠图 512×512 用时 21.5 s，输出 RGBA，透明像素 82.7%。
- **成片真出**：背景+角色合帧 1920×1080 → `h264_nvenc` + AAC 成片（demo 2.53 s / 215 KB），
  另一条三镜头 demo 11.51 s / 403 KB。
- 单元测试 25/25 通过。

## [1.2.0] - 2026-11-08

### Added

- `theater/hardware.py`：显卡探测与选型（纯标准库，无第三方依赖）。探测 `nvidia-smi` +
  `Win32_VideoController`（Windows）/ `lspci`（Linux），归一成 `nvidia` / `amd` / `intel`，
  按"独显优先"判定主显卡。
- `hardware.select_encoder`：按厂商从 `ffmpeg -encoders` 里挑真实可用的编码器
  （NVIDIA→`h264_nvenc`，AMD→`h264_amf`，Intel→`h264_qsv`），一个没有就落 `libx264`。
- `hardware.encoder_quality_args`：三家硬编码器参数分别拼——QSV 用 `-global_quality`、
  NVENC 用 `-rc vbr -cq`、AMF 用 `-rc cqp -qp_i/-qp_p`、软编用 `-crf`。
- `hardware.pick_draw_backend` + `suggest_openvino_device`：NVIDIA→torch/cuda、
  AMD→torch/dml（Windows）、Intel→openvino/GPU、其余→openvino/CPU。
- `tests/test_hardware.py`：19 个单测锁定上述决策（含"NVENC 绝不能收到 `-crf`"这类硬约束）。
- 新环境变量：`GPU_ADAPTERS` / `GPU_VENDOR`（覆盖探测）、`DRAW_BACKEND` / `DRAW_DEVICE`、
  `NVENC_CQ` / `AMF_QP`（各厂商画质）。

### Changed

- `theater/ai_draw.py`：从"仅 OpenVINO"改为**双后端**（OpenVINO / PyTorch）。PyTorch 走
  `diffusers.DiffusionPipeline`，设备分 CUDA / DirectML / CPU 三档。仍保留懒加载、线程锁、
  缓存命中跳过、离线抠图降级。
- `theater/video_export.py`：`VIDEO_ENCODER` 默认值由写死 `h264_qsv` 改为 `auto`；
  编码器解析改走 `hardware.select_encoder`，参数改走 `hardware.encoder_quality_args`；
  新增 `VIDEO_CODEC`（h264 / hevc / av1）。
- `check_env.py`：新增"显卡 / 主显卡厂商 / 绘图后端"三项；编码器检查从"只测 h264_qsv"
  改为"按厂商选定编码器并**真编 2 帧黑场**验证"，避免"有名字没实现"的假可用。
- `requirements.txt`：补 CUDA / DirectML / ROCm 三套可选安装命令与 ffmpeg 构建要求。
- `README.md`：新增第 11 节「显卡适配」——自动选型矩阵、模型格式对照、环境变量表、本机实测。

### Fixed

- 给非 Intel 平台传 `-crf` 被忽略或报错的问题（现按编码器分别传参）。
- 有独显却按核显判定设备的问题（现 NVIDIA > AMD > Intel，独显优先）。

## [1.1.0] - 2026-11-08

### Changed

- **授权模型改为双许可**：源代码可采用 **GNU AGPL-3.0-only**（`LICENSE`）或**商业许可**（`COMMERCIAL-LICENSE.md`），二选一。SPDX 表达式统一为 `AGPL-3.0-only OR LicenseRef-Commercial`。
- `LICENSE` 换为 AGPL-3.0 官方全文（34,523 字节，`https://www.gnu.org/licenses/agpl-3.0.txt` 原样落盘，未改一字）。
- 原 Apache-2.0 全文移至 `LICENSES/Apache-2.0.txt`，仅用于 v1.0.0–v1.0.1 历史版本（该授权不可撤回）。
- 全部 10 个源文件（9 个模块 + `tests/test_script_parser.py`）头部注释改写：SPDX 行、Apache 样板段换成「双许可 + AGPL 第 13 条提示」。
- `README.md` 顶部许可行、§9、§10 表格同步为三层授权结构（双许可 / EULA / 历史 Apache）。

### Added

- `LICENSING.md`：双许可选型说明 + 8 条 FAQ（单机自用、发视频、内网部署、闭源集成、只改剧本、fork、输出内容归属、能否换许可）。
- `COMMERCIAL-LICENSE.md`：可签署的商业许可协议模板（14 条，含授权范围、限制、费用栏位、输出内容责任、补偿、期限与终止）。
- `LICENSES/AGPL-3.0.txt`：AGPL-3.0 官方文本留档。

### Docs

- `NOTICE`：改为双语双许可声明 + 历史 Apache 授权说明。
- `EULA.md` 升版至 **v1.1**：第 8 条重写为「双许可 / 与 AGPL 的边界 / 商业许可优先 / 历史版本」四款。

> 为什么不采用「Apache-2.0 + 商业」双许可：Apache 已允许商用与闭源，再卖商业许可没有商业意义；双许可的价值来自 copyleft（详见 `LICENSING.md` 第 4 节）。

## [1.0.1] - 2026-11-08

### Added

- `EULA.md`：最终用户许可协议。源代码仍为 Apache-2.0；就“官方发行物 / 模型权重 / 品牌素材 / 署名”另行约定使用与再分发边界。
- EULA 第 4 条独立成章：**所有输出内容与作者无关**，作者未参与、未审核、未控制生成过程；因侵犯他人肖像权、声音权、名誉权、著作权、商标权，或传播违法信息、未按《人工智能生成合成内容标识办法》（2025-09-01 起施行）添加标识等引发的投诉、索赔、行政处罚与刑事追责，**全部由使用者承担**。
- EULA 第 9.2 条补偿条款：第三方主张指向作者时，使用者须全额补偿（含律师费、和解金、赔偿金）。
- `tests/test_script_parser.py`：7 个纯标准库单测（补上首镜头前正文与 `【镜头1】` 的合并逻辑）。

### Fixed

- `script_parser`：修复“镜头前正文”与首个 `【镜头1】` 生成两个同号镜头、导致帧文件互相覆盖的问题。

### Docs

- `README.md` §10：EULA 与 Apache-2.0 的边界对照表。
- `NOTICE`：补 EULA 引用与“生成内容与作者无关”声明。

## [1.0.0] - 2026-11-07 09:31:28 +08:00

首个可用版本：本地图文小剧场自动化（OpenVINO / Intel 平台），经 MCP 暴露给 LobeChat 分步调用。

### Added

- `theater/script_parser.py`：剧本结构化解析。支持 `【镜头N】`、`【角色：站位：左/右/居中】`、
  `角色：台词`、`【旁白】文本` 四种写法；元信息（场景/地点/背景…）不会被误判为台词；
  未写站位的角色按人数自动分配位置；旁白参与字幕与配音但不占站位、不出立绘。
- `theater/ai_draw.py`：OpenVINO 核显绘图（FLUX 优先 `OVFluxPipeline`，回退 SD pipeline），
  懒加载 + 线程锁；`rembg` 走离线 `U2NET_HOME`，模型缺失自动降级白底连通域抠图。
- `theater/local_tts.py`：SAPI5 离线配音，**每句独立 engine**（修复 pyttsx3 常见 0 字节问题），
  带 0 字节校验与一次重试；支持按角色指定发音人（`TTS_VOICE_MAP`）。
- `theater/frame_compose.py`：RGBA 半透明字幕条（修复 RGB 图上写 4 元组颜色报错）、
  字幕按像素宽度折行最多 3 行、同站位多角色自动均分铺开、帧缓存按签名命中。
- `theater/video_export.py`：QSV 用 `-global_quality`（修复 `-crf` 不被识别）、
  画面 `scale+pad` 到 1920×1080 不再变形、强制 `yuv420p`、音频统一 48kHz 重编码、
  拼接后时长校验并对不上时重编码兜底、编码器不可用自动回退 `libx264`。
- `server.py`：MCP stdio 服务，8 个工具（check_env / parse_script / generate_script /
  generate_assets / generate_tts / compose_frames / export_video / run_all）。
- `pipeline_full.py`：命令行一键流水线，支持 `--steps` 分步与 `--requirement` 调本地 LLM 写剧本。
- `check_env.py`：环境自检（位数/依赖/ffmpeg 编码器/模型/字体/发音人/缓存可写），FAIL/WARN 分级。
- `script.txt`：3 镜头示例剧本。

### Fixed

- 修复原始设计稿中 `server.py` / `check_env.py` / `requirements.txt` 三处缺失。
- 修复 `OVStableDiffusionPipeline` 误用于加载 FLUX、`local_files_only` 对本地目录无效。
- 修复所有 ffmpeg 调用忽略 returncode 导致静默失败。
- 缓存目录改为锚定项目根的绝对路径（MCP 由客户端拉起时工作目录不确定）。

### License

- Apache License 2.0，署名 MWnv1（GitHub: https://github.com/MWnv1）。
