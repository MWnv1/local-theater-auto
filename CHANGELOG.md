# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.5.1] - 2026-11-10

修 1.5.0 交付的"果冻"：参考视频 → 相似视频整幅变成大理石纹/融化块。
三处根因同时叠加才出这个效果，单改一处都不够。

### Fixed

- **去噪步数不足（主因）**：diffusers 的 img2img 实际只跑
  `int(num_inference_steps * strength)` 步。原 `SIM_STEPS=8` 配 `SIM_STRENGTH=0.35`
  只剩 **2 步**，等于拿噪声直接出图。`_restyle()` 现在强制
  `steps*strength ≥ SIM_MIN_DENOISE`（默认 12），传 8 步会被抬到 35 步。
- **重绘尺度太小**：`SIM_SIZE=512` 在 16:9 上只有 512×288，细节被抹平后再 LANCZOS
  放回 1080p 就是一片糊。改用 `768`（同参数对照：边缘 IoU 0.5749 vs 0.4449）。
- **上一帧生成图作 init 累积瑕疵**：`SIM_MOTION_WARP` 默认从 `1` 改为 `0`，
  每帧都从原片起手；想连续画风再手动开 `1`。
- 新增 `_guard_blend()`：出图后与原帧算边缘 IoU，低于 `SIM_MIN_IOU`（0.22）就按
  `SIM_GUARD_MIX`（0.55）混回原帧 —— 兜底"宁可保守也不交糊块"。

### Added

- 固定种子 `SIM_SEED`（默认 20261110），逐帧同一噪声源，消除同场景细节乱跳。
- **Canny ControlNet 分支（已实测启用）**：`SIM_CONTROLNET=auto|on|off`、
  `SIM_CONTROLNET_DIR`、`SIM_CONTROLNET_SCALE`（默认 0.9）。权重
  `lllyasviel/control_v11p_sd15_canny`（fp16，722,598,642 B，sha256
  `9ed3a7f8…c0bc` 已校验）放在 `models/controlnet-sd15-canny/`；存在时自动改用
  `StableDiffusionControlNetImg2ImgPipeline`。canny 图由 `_canny_edges()` 用 PIL + numpy
  现算（不装 opencv）：先按直方图众数判"纯色无边缘"，再用 5%/99% 分位归一化 + 阈值，
  `MaxFilter(3)` 加粗。实测额外加载约 60 s、每帧 +0.2 s，同一帧边缘 IoU 0.451（无 CN 是 0.414）。
  权重缺失或显式 `off` 时打印警告并回落普通 img2img。
- 新环境变量：`SIM_MIN_DENOISE`、`SIM_SEED`、`SIM_GUARD`、`SIM_MIN_IOU`、
  `SIM_GUARD_MIX`、`SIM_CONTROLNET`、`SIM_CONTROLNET_DIR`、`SIM_CONTROLNET_SCALE`。
- `theater/video_ref.py` 模块 docstring 记录"2026-11-10 果冻事故"的完整复盘。
- `tests/test_sim_quality.py`：9 条回归（默认值就是修好的那组、步数下限、固定种子、
  guard 把纯色输出混回原帧、好图不被改、边缘 IoU 能识别垃圾、canny 黑白图与纯色全黑、
  CN 权重齐备判定、CN 权重缺失回落）。
- 负向提示词补 `melting / warped / smeared / deformed face / out of focus / low contrast /
  grainy / noisy / speckles / dithering / dust`。

### Changed

- **边缘 IoU 指标改成尺度无关**：`_edge_mask()` 先缩到 768 长边再取边缘，否则同一对
  1080p 图只有 0.08、缩到 768 却有 0.41，数字没法比。改完的统一口径：果冻版
  **0.0805** → 修复版 **0.4298**（8 帧均值，纯 img2img）/ **0.4281**（带 ControlNet）。
- 修复后实测（8 帧全画面平移参考）：运动判定仍是 `global` 向左 84 px/帧、
  生成片 82.5 px/帧，`direction_match=true`、`speed_ratio=1.009`、`warped_frames=0`；
  整条 124 s 左右（2.0~2.2 s/帧，权重加载另计 ~60~100 s）。
- `_controlnet_on()` 默认强度回到 0.35：实测 0.45 会在墙面/天花板刷出颗粒噪点，
  结构的收益在 0.35 已经吃满。
- README §13.1.2 / §13.3 / §13.4 / §13.5 同步（步数下限、尺度、guard、ControlNet 与踩坑表）。
- 测试总数 83 → 92（83 + 9）。

## [1.5.0] - 2026-11-10

加入 CUDA / Tensor Core 调用与"按本机配置"的量化。默认行为更省显存、更快，
且全部可关（`ACCEL=off` 退回 torch 默认）。

### Added

- `theater/accelerator.py`（新模块）：硬件画像（GPU / 显存 / sm 计算能力 / torch / CUDA /
  CPU / RAM）、按显存分档（`xl` ≥24GB、`high` ≥12GB、`mid` ≥8GB、`low`、`cpu`）、
  `apply_global()` 打开 TF32 与 `cudnn.benchmark`、`_pick_attention()` **实跑**探测
  SDPA 内核（flash → mem_efficient → math）、`llm_kwargs()` 返回带
  `BitsAndBytesConfig`（NF4 4bit + double quant，8bit 可选）的加载参数、
  `sd_kwargs()` / `optimize_pipe()`（channels_last）、`tensor_tflops()` 实测吞吐、
  `describe()` / `status()`。
- 新环境变量：`ACCEL`、`ACCEL_LLM_QUANT`、`ACCEL_SD_DTYPE`、`ACCEL_TF32`。
- `tests/test_accelerator.py`：10 条单测（档位与显存一致、报出来的 attention 内核真能跑、
  `apply_global()` 幂等且 `float32_matmul_precision=high`、GPU 上 `llm_kwargs()` 必带量化配置、
  `ACCEL=off` 全关、Tensor Core 实测为正、`optimize_pipe()` 容错）。
- `check_env.py` 新增两项："GPU 加速/量化档位"（打印 GPU/显存/sm/档位/精度/内核/TF32/LLM 量化）
  与"Tensor Core 实测（4096³ matmul）"（fp32/fp16/bf16 的 TFLOP/s）。

### Changed

- `pipeline_full.py::_load_llm`：改用 `accelerator.llm_kwargs()`，剧本模型默认按档位走
  NF4 4bit（Qwen3-4B 显存 7.49 GB → **2.48 GB**），加载前打印量化说明。
- `theater/video_ref.py::_get_pipe`：dtype 与 attention 内核由 accelerator 给，新增
  `channels_last`，日志打印实际生效项。
- `theater/ai_draw.py::_load_torch_pipe`：不再硬编码 bf16，按档位取精度并先 `apply_global()`。
- `theater/tts_clone.py::_ensure_f5`：F5 加载前 `apply_global()`，fp32 权重吃 TF32。
- `README.md` 新增 §14（档位表、四条接线、实测表、踩坑），§13.7 测试数改 83。

### Fixed

- 探测 attention 内核时 torch 会为每个不可用内核刷 6 行 `UserWarning`，用
  `warnings.catch_warnings()` 压掉，避免每次导入都污染日志。
- `check_env.py` 的 Tensor Core 实测原本在 `apply_global()` 之前跑，fp32 报出的是
  "没开 TF32"的数（16.7 TFLOP/s）；现在先开 TF32 再测（24.5 TFLOP/s）。

## [1.4.1] - 2026-11-10

人物动作收窄成"只做左右平移 / 上下平移 / 旋转"；参考视频 → 相似视频改成
"逐帧算偏移 → AI 分析 → 再生成"。两条都是行为变更，接口保持向后兼容。

### Changed

- `theater/animator.py` **整文件重写**：放弃部件级伪骨骼旋转，改成整图仿射。
  `ACTIONS` 新增 `pan_left` / `pan_right` / `pan_up` / `pan_down`（周期 4.0 s，
  幅度 ±0.18 图宽 / ±0.16 图高）、`rotate`（±10°，周期 3.0 s）、`spin`（360°/6.0 s）；
  旧名 `idle` / `nod` / `wave` / `walk` / `talk` 保留成**小幅度平移/旋转的别名**
  （amp_y 0.012 / 0.03、amp_x 0.05 / 0.09、rot 1.6°），老脚本与 `build_anims` 不用改。
- `pose_at(action, t)` 返回值改为归一化 `{dx, dy, rot}`（dx/dy 占图宽高的比例），仍是纯函数；
  新增 `transform_px(pose, size)` 换算像素，新增 `warp(img, dx, dy, rot)` 用一次性
  `Image.AFFINE`（绕中心旋转 + 平移，BICUBIC，透明填充）出帧，避免"先 rotate 再 paste"的
  取整抖动与边缘拉丝。
- `render_clip()` manifest 新增 `motion_tracks`（每帧 `{t, pose, px}`），
  保留 `joint_tracks` 键指向同一份数据以兼容旧断言。
- `theater/video_ref.py` 新增逐帧运动分析整段：`frame_offset(prev, cur)`（缩到
  `SIM_MOTION_SIZE` 灰度 → 去均值 + 汉宁窗 → `rfft2` 相位相关 → `(dx,dy)` 与 `(-dx,-dy)`
  两组都算重叠区相关系数取更像的 → ±8°、步长 2 搜旋转再 ±1° 精修）、`describe_motion()`、
  `analyze_motion()`、`motion_hint_llm()`（可选接本地 Qwen3-4B）、`_warp_prev()`、
  `motion_signature()`。`frame_offset` 顺带返回 `changed`（像素变化比例）。
- `generate_similar_video()` 主流程串入：抽帧 → `analyze_motion()` → `[motion]` 日志 →
  `motion_hint_llm()` 取提示（失败回落规则提示）→ `prompt_all = "<提示>, <运动提示>"` →
  逐帧重绘；仅当 `kind=global`、位移 ≥2 px（或旋转 ≥0.8°）且 `conf` ≥0.35 时，才把上一帧
  生成图按本帧偏移推成 init（`SIM_MOTION_WARP`），`local`/`static` 一律回用原帧。
  `workdir\motion.json` 落 summary / llm_hint / prompt_used / 全部 offsets；
  `out_info["motion"]` 加 `kind` 与 `warped_frames`。
- `similarity_report()` 增加运动一致性：两段视频各抽帧做同样分析，比较方向点积
  （`direction_match`，两边都近乎静止也算一致）、`speed_ratio`、各自累计旋转；
  失败只写 `report["motion"]["error"]`，不影响时长/帧率/分辨率/边缘 IoU。
- `pipeline_full.build_anims()`：发言角色由 `talk` 改为 `rotate`（新规格下
  "说话时轻轻转动"比 1.6° 更明显），其余角色仍 `idle` / `nod`。
- `server.py`：`theater_animate_character` 文档串与返回体更新（动作列表 + `motion` 摘要）。
- `check_env.py`：检查项改名"整体位移/旋转动作"，新增"逐帧运动分析"（分析尺度 /
  AI 分析开关 / 运动作 init 开关）。
- `README.md` §13：新增 13.1.1（平移/旋转动作表）与 13.1.2（逐帧偏移 → AI 分析 → 生成
  的五步顺序）、`SIM_MOTION_*` 三个环境变量、校准与端到端实测数字、三条新踩坑。

### Added

- `tests/test_motion.py`：20 条（动作注册与方向、幅度互为相反数、旋转随时间增大、
  `spin` 角度范围、`pose_at` 纯函数、`transform_px` 换算、`warp` 平移/旋转能被
  `frame_offset` 同号复现、`analyze_motion` 逐对偏移、向右/静止/顺时针三种归纳、
  `local` 与 `static` 分类、`changed` 比例、LLM 默认关、`pan_left` 两帧有差异、
  manifest 的 `motion_tracks`）。总数 53 → 73。
- `_tools/probe_motion.py`（已知位移校准）、`_tools/probe_motion_similar.py`
  （逐帧分析 + 相似视频端到端）、`_tools/probe_motion_kind.py`（local/global/static 分类）、
  `_tools/make_pan_ref.py` / `make_pan_ref_full.py` / `make_pan_show.py`（造参考视频）。

### Fixed

- 旋转角与 `warp()` 转向相反：PIL `rotate(角度)` 逆时针，而 `warp()` 按图像坐标正方向
  顺时针，`_gray_rotated` 改用 `im.rotate(-deg)` 后，已知 +6° 能测回 +6.0°（conf 0.998）。
- 大面积静态背景把相位相关峰值压回 (0,0)，人物在画面里走动被测成 0 px：
  用 `changed` 比例区分 `local`（主体在动）与 `global`（镜头/整体在动），`local` 不再推整帧。
- 上一帧生成图当 init 的误差累积（生成片自己漂移）：只在 `global` 且位移足够大时推。

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
- `server.py::_out_path()`：`clone_voice(out_name="probe")` 这种不带后缀的写法会一路传到
  pydub，再报 `No format specified and unable to get format from file extension` 并静默
  退回 SAPI——音频确实生成了，但不是克隆音色。改成按用途补默认后缀
  （音频 `.wav`、视频 `.mp4`）。
- `theater/video_ref.py::generate_similar_video()`：组装帧序列时同样撞上 concat demuxer
  不认 `-framerate`，改 `-r <fps>`。

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
