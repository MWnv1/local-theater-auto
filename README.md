# 本地图文小剧场自动化（OpenVINO / Intel 版）

> Copyright 2026 **MWnv1**（GitHub: https://github.com/MWnv1）
> 个人站点：https://imworkingnow.cc.cd ｜ https://imwn.cc.cd ｜ https://imwn.dpdns.org
> 许可：**Apache License 2.0**（见 `LICENSE`、`NOTICE`）
> 版本生成时间：**2026-11-07 09:31:28 +08:00**（Asia/Shanghai / UTC 2026-11-07T01:31:28Z / Unix 1794015088）

剧本 → 背景与角色立绘 → 台词配音 → 镜头画面（含字幕）→ 成片，全本地离线运行，
通过 MCP 暴露给 LobeChat 分步调用。

## 目录结构

```
D:\AI_T_MCP\
├─ server.py            # MCP Server（stdio），LobeChat 的入口
├─ pipeline_full.py     # 流水线 + 命令行入口（也可让本地 LLM 写剧本）
├─ check_env.py         # 环境自检
├─ requirements.txt
├─ README.md
├─ output\              # 成片输出
├─ cache\               # 素材/音频/帧/片段缓存（可删）
└─ theater\
   ├─ __init__.py       # 路径与公用工具（绝对路径锚定）
   ├─ script_parser.py  # 剧本结构化
   ├─ ai_draw.py        # OpenVINO 绘图 + 离线抠图
   ├─ local_tts.py      # SAPI5 离线配音
   ├─ frame_compose.py  # 图层合成 + 字幕
   └─ video_export.py   # QSV/NVENC/libx264 导出
```

## 1. 环境

- **64 位 Python 3.11**（强烈建议；OpenVINO/optimum-intel 官方 wheel 目前只到 3.12 左右，
  本机默认的 3.14 装不上 openvino）
- **ffmpeg + ffprobe**（建议带 `h264_qsv` 的静态构建），把 `bin` 加进 PATH，
  或设 `FFMPEG_PATH` / `FFPROBE_PATH`
- Intel 显卡/NPU 驱动最新

```cmd
cd D:\AI_T_MCP
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python check_env.py
```

> 仓库里的 `.venv` 是本次验证时用 3.14 建的，只装了 pillow / pyttsx3，**不要用它装 openvino**；
> 正式环境请按上面的 3.11 流程重建（或直接删掉把 `.venv` 重开）。

`check_env.py` 里 FAIL 必须先修；WARN 可以带病运行（会自动降级并打印告警）。

## 2. 模型摆放（默认 `D:\ai_models`，用 `MODEL_ROOT` 可改）

| 用途 | 目录 | 说明 |
|---|---|---|
| 绘图 | `D:\ai_models\flux-schnell-int4-ov` | Flux Schnell INT4 OpenVINO 预转换包 |
| 抠图 | `D:\ai_models\rembg\u2net.onnx` | 必须预置，否则**不会联网下载**，会降级为白底抠图 |
| 剧本 LLM（可选） | `D:\ai_models\qwen3-4b-int4-ov` | Qwen3-4B INT4 OpenVINO |

预转换包只要有 `openvino_model.xml/.bin` + `model_index.json` 之类即可；本项目
会优先用 `OVFluxPipeline` 加载 FLUX，旧版 optimum-intel 缺该类时回退
`OVStableDiffusionPipeline`。

## 3. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `MODEL_ROOT` | `D:\ai_models` | 模型根目录 |
| `THEATER_CACHE` | `<项目>\cache` | 缓存根目录 |
| `FFMPEG_PATH` / `FFPROBE_PATH` | PATH 中 | 可执行文件绝对路径 |
| `DRAW_MODEL_DIR` | `$MODEL_ROOT\flux-schnell-int4-ov` | 绘图模型 |
| `DRAW_DEVICE` | `GPU` | `GPU` / `CPU` / `AUTO` |
| `DRAW_STEPS` / `DRAW_SIZE` | `4` / `1024` | 采样步数、出图边长 |
| `VIDEO_ENCODER` | `h264_qsv` | 不可用时自动回退 `libx264` |
| `QSV_QUALITY` | `23` | QSV 的 `-global_quality` |
| `X264_CRF` | `20` | 回退 x264 时的 CRF |
| `VIDEO_W` / `VIDEO_H` / `VIDEO_FPS` | `1920/1080/30` | 输出规格 |
| `TTS_RATE` | `180` | 语速 |
| `TTS_VOICE` / `TTS_VOICE_MAP` | 空 | 发音人；映射例 `小林=Huihui,阿杰=Kangkang` |
| `LLM_MODEL_DIR` / `LLM_DEVICE` | `$MODEL_ROOT\qwen3-4b-int4-ov` / `NPU` | 剧本生成 |

## 4. 剧本格式

```
【镜头1】
【小林：站位：左】
【阿杰：站位：右】
小林：这个模型真能离线跑？
阿杰：4bit 的，NPU 上 30 token/s。

【镜头2】机房门口
【旁白】夜色渐深。
阿杰：走吧，明天继续调。
```

- `【镜头N】` 后可接场景描述（会用来单独出背景）
- `【角色：站位：左/右/居中】` 可省略，省略时按角色数自动分配
- `场景：xxx` 会被当成镜头元信息，不会误判成台词

## 5. 命令行

```cmd
python pipeline_full.py --script script.txt --out output\demo.mp4
python pipeline_full.py --script script.txt --steps parse          :: 只看解析结果
python pipeline_full.py --requirement "两个程序员在机房聊本地AI小剧场" --out output\demo.mp4
```

## 6. LobeChat MCP 配置

```json
{
  "mcpServers": {
    "local-theater-auto": {
      "command": "python",
      "args": ["D:\\AI_T_MCP\\server.py"],
      "env": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODEL_ROOT": "D:\\ai_models",
        "FFMPEG_PATH": "D:\\ai_tools\\ffmpeg\\bin\\ffmpeg.exe",
        "FFPROBE_PATH": "D:\\ai_tools\\ffmpeg\\bin\\ffprobe.exe"
      }
    }
  }
}
```

## 7. 本机（AMD + NVIDIA）跑这套代码的真实预期

这套代码按 Intel 平台写死，在你当前机器上：

- `h264_qsv` **不存在** → 导出会自动回退 `libx264`（慢一些，但能出片）
- `DRAW_DEVICE=GPU` 指的是 **Intel GPU**，AMD 核显/NVIDIA 都不适用 →
  绘图会报设备不可用。真要出图，需要换 CUDA/ComfyUI 后端（可用 `DRAW_DEVICE=CPU` 先跑通流程看效果，非常慢）
- `LLM_DEVICE=NPU` 需要 Intel NPU → 本机没有，写剧本这步不可用；剧本文本手写或外部模型给
- 合成、字幕、TTS（SAPI5）、拼接这几步与平台无关，可以正常跑

即：**流程、解析、字幕、配音、拼片可用；绘图与 NPU 推理需要 Intel 硬件。**

## 8. 本次已验证 / 未验证

已验证（本机实跑）：

| 项目 | 结果 |
|---|---|
| 全部 9 个 .py 字节码编译 | 通过（`py_compile` 无报错） |
| `script_parser` 解析 `script.txt` | 3 镜头、角色 小林/阿杰；旁白不占站位、`【旁白】xxx` 正确进入台词 |
| `frame_compose` 合成 | 输出 1920×1080 PNG；长句自动折 3 行 + 省略号；半透明字幕条生效（像素采样变暗）；3 角色同站"居中"自动均匀铺开不重叠 |
| `local_tts`（SAPI5） | 7 句全部产出非 0 字节 wav（81–181KB），无"只有第一句有声音"问题 |
| 时长解析 | 走 `wave` 兜底读出 2.50/4.11/3.03… 秒，不再回落 3.0 默认值 |
| `check_env` | 中文正常输出，FAIL/WARN 分级正确；检测到 2 个发音人（Huihui / Zira） |

未验证（缺依赖/模型，非代码问题）：`ai_draw` 绘图（无 OPENVINO + 无模型）、
`video_export` 实际编码（本机无 ffmpeg）、`server.py` 的 MCP 握手（未装 mcp 包）。

## 9. 许可与署名

```
Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
License:  Apache License, Version 2.0  (SPDX: Apache-2.0)
```

- `LICENSE`：Apache-2.0 官方全文（未做任何删改）
- `NOTICE`：署名、站点、生成时间戳与第三方组件许可清单
- 每个 `.py` 文件头部均带 `SPDX-License-Identifier: Apache-2.0` + 版权/署名/站点/时间戳注释块
- 再分发时按 Apache-2.0 第 4 条：保留 LICENSE 与 NOTICE，修改过的文件需标注"已修改"

> 提示：`script.txt` 顶部那几行 `#` 注释会被解析器自动跳过，不影响剧本解析。

## 10. 最终用户许可协议（EULA）

见 [`EULA.md`](EULA.md)。两层授权，边界如下：

| 对象 | 适用条款 |
|---|---|
| 源代码 | Apache License 2.0（`LICENSE`）——可自由使用、修改、商用、再分发 |
| 官方发行物 / 模型权重 / 品牌素材 / 署名 | EULA（`EULA.md`）——不得单独抽取再分发、不得去除署名、不得作为付费服务转售 |

**关于输出内容（EULA 第 4 条）**：本软件是通用本地工具，不含任何真实人物形象或他人素材。
你生成的一切内容与作者无关；由此产生的一切后果（含侵犯他人**肖像权、声音权、名誉权、
著作权、商标权**，或传播违法信息、未按《人工智能生成合成内容标识办法》加标识等），
**全部由使用者自行承担**，作者不承担直接、连带或补充责任（详见 EULA 第 4、7、9 条）。
