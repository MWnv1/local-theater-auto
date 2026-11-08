# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
