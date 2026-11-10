# 本地图文小剧场自动化（NVIDIA / AMD / Intel 通用）

> Copyright 2026 **MWnv1**（GitHub: https://github.com/MWnv1）
> 个人站点：https://imworkingnow.cc.cd ｜ https://imwn.cc.cd ｜ https://imwn.dpdns.org
> 许可：**AGPL-3.0-only 或 商业许可（双许可，二选一）**（见 `LICENSE`、`LICENSING.md`、`COMMERCIAL-LICENSE.md`）
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
   ├─ hardware.py       # 显卡探测（NVIDIA/AMD/Intel）→ 后端与编码器选型
   ├─ script_parser.py  # 剧本结构化
   ├─ ai_draw.py        # 绘图（OpenVINO / PyTorch 双后端）+ 离线抠图
   ├─ local_tts.py      # SAPI5 离线配音
   ├─ frame_compose.py  # 图层合成 + 字幕
   └─ video_export.py   # NVENC / AMF / QSV / libx264 导出
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
License:  AGPL-3.0-only  OR  Commercial License  (choose one)
SPDX:     AGPL-3.0-only OR LicenseRef-Commercial
```

- `LICENSE`：GNU AGPL-3.0 官方全文（`https://www.gnu.org/licenses/agpl-3.0.txt`，未做任何删改）
- `LICENSING.md`：双许可说明与选型 FAQ（什么情况必须买商业许可）
- `COMMERCIAL-LICENSE.md`：商业许可协议模板（可签署）
- `LICENSES/Apache-2.0.txt`：历史版本（v1.0.0–v1.0.1）所用 Apache-2.0 全文，**该授权不可撤回**
- `NOTICE`：署名、站点、生成时间戳与第三方组件许可清单
- 每个 `.py` 文件头部均带 `SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial` + 版权/署名/站点/时间戳注释块
- 走 AGPL 通道时：分发本软件、或把它作为网络服务对外开放，都必须向使用者提供你修改版本的完整对应源码（AGPL 第 13 条）

> 提示：`script.txt` 顶部那几行 `#` 注释会被解析器自动跳过，不影响剧本解析。

## 10. 最终用户许可协议（EULA）

见 [`EULA.md`](EULA.md)。授权分三层，边界如下：

| 对象 | 适用条款 |
|---|---|
| 源代码 | **AGPL-3.0-only**（社区版，`LICENSE`）**或** 商业许可（`COMMERCIAL-LICENSE.md`），二选一 |
| 官方发行物 / 模型权重 / 品牌素材 / 署名 | EULA（`EULA.md`）——不得单独抽取再分发、不得去除署名、不得作为付费服务转售 |
| 历史版本（v1.0.0–v1.0.1） | Apache-2.0（`LICENSES/Apache-2.0.txt`），不可撤回 |

**双许可怎么选**：个人自用、发视频、内部部署 → 直接按 AGPL-3.0 用，无额外义务；
闭源集成、私有化交付给客户、对外做 SaaS → 买商业许可。详细判定见 [`LICENSING.md`](LICENSING.md)。

**关于输出内容（EULA 第 4 条）**：本软件是通用本地工具，不含任何真实人物形象或他人素材。
你生成的一切内容与作者无关；由此产生的一切后果（含侵犯他人**肖像权、声音权、名誉权、
著作权、商标权**，或传播违法信息、未按《人工智能生成合成内容标识办法》加标识等），
**全部由使用者自行承担**，作者不承担直接、连带或补充责任（详见 EULA 第 4、7、9 条）。

## 11. 显卡适配（NVIDIA / AMD / Intel）

代码会自动探测本机显卡，决定**绘图后端**与**视频编码器**，无需手工改参数。
探测逻辑在 `theater/hardware.py`，纯标准库，探测失败只降级不报错。

### 11.1 自动选型矩阵

| 你的显卡 | 绘图后端（需自行 pip 装） | 视频编码器 | 参数差异 |
|---|---|---|---|
| **NVIDIA**（RTX / GTX / Quadro） | `torch` + **CUDA**（最快） | `h264_nvenc`（`hevc_nvenc` / `av1_nvenc`） | `-rc vbr -cq N -b:v 0` |
| **AMD**（RX 6000/7000、核显） | Windows：`torch` + **DirectML**；Linux：ROCm 版 torch | `h264_amf`（`hevc_amf` / `av1_amf`） | `-rc cqp -qp_i N -qp_p N` |
| **Intel**（核显 / Arc） | `openvino` + **GPU** | `h264_qsv`（`hevc_qsv` / `av1_qsv`） | `-global_quality N` |
| 无独显 / 无驱动 / 无 ffmpeg | `openvino` + **CPU** | `libx264` | `-crf N` |

**注意 `-crf` 不是通用的**：原稿给 QSV 传 `-crf` 会被忽略，给 NVENC/AMF 传更是直接报错。
现在每个编码器的画质旋钮按上表分别拼参（`theater/hardware.py::encoder_quality_args`），
由单测锁死，见 `tests/test_hardware.py`。

### 11.2 装哪套后端

```bash
# Intel 核显 / Arc / 纯 CPU（基线，任何平台都能跑）
pip install openvino openvino-tokenizers "optimum[openvino]" nncf diffusers

# NVIDIA（CUDA 版 PyTorch；RTX 50 系 / Blackwell 要 cu128 及以上）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# AMD（Windows 用 DirectML；Linux 换 ROCm 版 torch）
pip install torch torch-directml
```

三者都不冲突：装了 torch 又有 NVIDIA → 自动走 cuda；AMD 机器只有 DirectML 可用；
Intel 机器没有 DirectML 就回 OpenVINO GPU。

### 11.3 模型格式要对上

| 后端 | 模型目录里应该是什么 |
|---|---|
| `openvino` | OpenVINO IR（`*.xml` + `*.bin`），如 `flux-schnell-int4-ov` |
| `torch` | HuggingFace 格式（`model_index.json` + `*.safetensors`），如 `FLUX.1-schnell` |

INT4 的 OpenVINO 模型**不能**给 torch 用，反之亦然。`DRAW_MODEL_DIR` 指向哪个目录，
就要配对应后端（`DRAW_BACKEND=auto` 时按显卡选，选错会给出明确报错而不是崩）。

### 11.4 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `DRAW_BACKEND` | `auto` | `auto` / `openvino` / `torch` |
| `DRAW_DEVICE` | `auto` | `auto` / `GPU` / `CPU` / `NPU` / `cuda` / `cuda:1` / `dml` |
| `DRAW_MODEL_DIR` | `$MODEL_ROOT/flux-schnell-int4-ov` | OpenVINO IR 绘图模型目录 |
| `TORCH_MODEL_DIR` | `$MODEL_ROOT/flux-schnell` | torch 后端用的 diffusers 模型目录 |
| `VIDEO_ENCODER` | `auto` | `auto` 或强制指定 `h264_nvenc` / `h264_amf` / `h264_qsv` / `libx264` |
| `VIDEO_CODEC` | `h264` | `h264` / `hevc` / `av1` |
| `NVENC_CQ` | `23` | NVENC 画质（0–51，越小越好） |
| `AMF_QP` | `22` | AMF 画质 |
| `QSV_QUALITY` | `23` | QSV 画质 |
| `X264_CRF` | `20` | 软编画质 |
| `GPU_ADAPTERS` | — | 手动覆盖探测结果，如 `NVIDIA GeForce RTX 5060 Ti\|AMD Radeon(TM) Graphics` |
| `GPU_VENDOR` | — | 只覆盖厂商判定：`nvidia` / `amd` / `intel` |

### 11.5 本机实测（RTX 5060 Ti + Ryzen 5 9600X 核显，已全量装到 D 盘）

```
[OK  ] 显卡              NVIDIA GeForce RTX 5060 Ti、AMD Radeon(TM) Graphics
[OK  ] 主显卡厂商        nvidia
[OK  ] 绘图后端          torch / cuda（NVIDIA 显卡 + PyTorch CUDA）
[OK  ] ffmpeg            D:\AI_T_MCP\tools\ffmpeg\bin\ffmpeg.exe
[OK  ] 选定视频编码器    h264_nvenc（nvidia 平台，硬编）
[OK  ] h264_nvenc 实际可用性   编码测试通过
[OK  ] 绘图模型          D:\ai_models\flux-schnell（diffusers 目录）
[OK  ] 抠图模型 u2net.onnx     D:\ai_models\rembg\u2net.onnx
```

- 探测正确：同时识别独显与核显，按"独显优先"判成 `nvidia`；
- 六种编码器**真编过**（各编 2–3 帧黑场）：`h264_nvenc` / `hevc_nvenc` / `av1_nvenc` / `h264_amf` / `hevc_amf` / `libx264` 全部通过；
  `h264_qsv` 只有本机是 Intel 核显/Arc 时才可能真跑，在 AMD 核显上会失败并被跳过；
- 绘图：CUDA 版 torch（2.9.1+cu128）+ FLUX.1-schnell 的 diffusers 权重 →
  `DRAW_BACKEND=auto` 自动落到 `torch / cuda`；INT4 OpenVINO 模型（`.xml/.bin`）不能给 torch 用。
- 装法与路径见下一节。


## 12. 全 D 盘安装（本机已落地，可照抄）

目标：**C 盘一个字节都不放**——Python、虚拟环境、ffmpeg、模型、pip 缓存/临时目录全在 D 盘。
下面的路径就是本机实测可用的最终形态。

### 12.1 装完的目录结构

```
D:\
├─ Python312\                        Python 3.12.10（64 位，含 pip，不进 PATH）
└─ AI_T_MCP\
   ├─ .venv312\                      项目虚拟环境（3.12，不提交）
   ├─ tools\ffmpeg\bin\              ffmpeg.exe / ffprobe.exe / ffplay.exe
   ├─ _tools\                        安装垃圾场：安装包、wheel、下载脚本、pip 缓存、临时目录（不提交）
   └─ theater\ ...                   项目源码
D:\ai_models\
├─ flux-schnell\                    FLUX.1-schnell 的 diffusers 权重（约 31.4 GB）
└─ rembg\u2net.onnx                 离线抠图模型（176 MB）
```

为什么不用系统里那个 Python：本机 `py -3` 是 3.14，而 **OpenVINO / onnxruntime / torch 都没有 3.14 的轮子**，
装依赖会在解析阶段就失败。所以要单独装一个 3.12，并且只服务这个项目。

### 12.2 五步装完（PowerShell，可整段粘）

```powershell
# 1) Python 3.12 静默装到 D 盘（华为云镜像，约 8 MB/s）
curl.exe -L -o D:\AI_T_MCP\_tools\python-3.12.10-amd64.exe `
  https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe
& D:\AI_T_MCP\_tools\python-3.12.10-amd64.exe /quiet InstallAllUsers=0 `
  TargetDir=D:\Python312 Include_pip=1 Include_launcher=0 PrependPath=0 `
  Shortcuts=0 Include_test=0 Associates=0

# 2) 建虚拟环境（3.12），并把 pip 的缓存/临时目录也挪到 D 盘
D:\Python312\python.exe -m venv D:\AI_T_MCP\.venv312
$env:PIP_CACHE_DIR = "D:\AI_T_MCP\_tools\pipcache"
$env:TEMP = "D:\AI_T_MCP\_tools\temp"; $env:TMP = $env:TEMP
D:\AI_T_MCP\.venv312\Scripts\python.exe -m pip install -U pip setuptools wheel `
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 3) 装 CUDA 版 PyTorch（RTX 50 系必须 cu128；这里走阿里云 wheel 镜像）
curl.exe -L -C - -o "D:\AI_T_MCP\_tools\torch-2.9.1+cu128-cp312-cp312-win_amd64.whl" `
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"
D:\AI_T_MCP\.venv312\Scripts\python.exe -m pip install --no-deps --force-reinstall `
  "D:\AI_T_MCP\_tools\torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"

# 4) 其余依赖（分批装，一次装完会撞 pip 的 resolution-too-deep）
$py = "D:\AI_T_MCP\.venv312\Scripts\python.exe"
& $py -m pip install numpy"<"2.3 pillow pyttsx3 comtypes -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
& $py -m pip install openvino openvino-tokenizers "optimum[intel]" nncf -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
& $py -m pip install transformers diffusers accelerate safetensors huggingface_hub -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
& $py -m pip install onnxruntime rembg mcp -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 5) ffmpeg（要带 nvenc/amf/qsv 的静态构建）
curl.exe -L -C - -o D:\AI_T_MCP\_tools\ffmpeg-btbN.zip `
  "https://gh-proxy.com/https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
Expand-Archive -LiteralPath D:\AI_T_MCP\_tools\ffmpeg-btbN.zip -DestinationPath D:\AI_T_MCP\tools\ffmpeg_tmp -Force
Move-Item D:\AI_T_MCP\tools\ffmpeg_tmp\ffmpeg-master-latest-win64-gpl D:\AI_T_MCP\tools\ffmpeg
```

装完后的 ffmpeg 会被自动找到：`theater/__init__.py::ffmpeg_path()` 依次找
`FFMPEG_PATH` → `项目根\tools\ffmpeg\bin` → `项目根\tools\ffmpeg` → `_tools\ffmpeg\bin` → `_tools\ffmpeg`
→ `D:\ffmpeg\bin` → `D:\ffmpeg` → `PATH`。所以哪怕不加环境变量也能直接跑。

### 12.3 模型（都放 D:\ai_models）

```powershell
# NVIDIA/AMD（torch 后端）：FLUX.1-schnell 的 diffusers 权重，约 31.4 GB
# HF 那个仓库需要登录，国内用 ModelScope 的镜像更快更省事
D:\AI_T_MCP\.venv312\Scripts\python.exe D:\AI_T_MCP\_tools\mdl.py `
  "https://modelscope.cn/models/black-forest-labs/FLUX.1-schnell/resolve/master" `
  "D:\ai_models\flux-schnell" "D:\AI_T_MCP\_tools\flux_manifest.txt" 20 --verify

# 抠图模型 u2net（离线，rembg 会读 U2NET_HOME 指向的 MODEL_ROOT\rembg）
D:\AI_T_MCP\.venv312\Scripts\python.exe D:\AI_T_MCP\_tools\pdl.py `
  "https://modelscope.cn/models/AI-ModelScope/u2net/resolve/master/u2net.onnx" `
  "D:\ai_models\rembg\u2net.onnx" 8
```

`mdl.py` / `pdl.py` 都在 `_tools\`：前者按清单批量多线程下整个模型目录，后者单文件多线程分段下。
断点续传看 `D:\ai_models\flux-schnell\.mdl_state.txt`（每下完一块追加一行 `路径:块号`），
**只有状态里记过的块才跳过，不看文件大小**——多线程下载会先把文件 truncate 到满尺寸，
按大小判断会把零填充的空壳当成"已下完"，直到 safetensors 加载才炸。
`--verify` 会在下完后逐块统计全零块占比，任一文件 < 99% 就输出 `VERIFY FAILED`。
单线程 curl 在这台机器上只有 400 KB/s，16–20 线程能到 9 MB/s，2.86 GB 的 torch 和 31.4 GB 的 FLUX 都靠它才下得动。

### 12.4 验证（四条，全过才算装好）

```powershell
cd D:\AI_T_MCP
D:\AI_T_MCP\.venv312\Scripts\python.exe check_env.py                 # 期望 FAIL 0 项
D:\AI_T_MCP\.venv312\Scripts\python.exe -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
D:\AI_T_MCP\.venv312\Scripts\python.exe -m unittest discover -s tests  # 期望 25/25 OK
D:\AI_T_MCP\tools\ffmpeg\bin\ffmpeg.exe -hide_banner -encoders | findstr nvenc
```

本机实测结果：

```
Python 3.12.10 64-bit          OK
torch 2.9.1+cu128              cuda True / NVIDIA GeForce RTX 5060 Ti / 15.9 GB / bf16 True
ffmpeg N-128691 (2026-11-08)   h264_nvenc / hevc_nvenc / av1_nvenc / h264_amf / hevc_amf / libx264 真编全通过
check_env.py                   FAIL 0 项，WARN 2 项（videotoolbox 不适用、Qwen3 剧本模型未下）
单元测试                        25/25 OK
```

### 12.5 网络坑（本机实测，别踩第二遍）

| 目标 | 结果 |
|---|---|
| `python.org` 直连 | HEAD 200，但下到 2 MB 就 `SEC_E_DECRYPT_FAILURE` → 换华为云镜像 |
| `download.pytorch.org` 直连 | 12 s 超时 / TLS `DECRYPTION_FAILED_OR_BAD_RECORD_MAC` → 换阿里云 `pytorch-wheels` |
| `huggingface.co` | 直连超时；FLUX 还是 gated 仓库，匿名 403 → 换 ModelScope |
| `www.gyan.dev` 的 ffmpeg | 12 KB/s，两小时下不完 → 换 `gh-proxy.com` 拉 BtbN 的 GPL 构建 |
| `ghproxy.net` / `hub.gitmirror.com` | 6 KB/s 超时 / DNS 失败 |
| `mirrors.aliyun.com/pytorch-wheels/cu128/` | 单线程 400 KB/s，16 线程 9 MB/s（可用） |
| `mirrors.tuna.tsinghua.edu.cn` | pip 主源，稳定 |
| `modelscope.cn` | 单线程 400 KB/s，多线程 9 MB/s（可用） |

### 12.6 FLUX 在 16 GB 显存上怎么跑

FLUX.1-schnell 是 12B 参数，bf16 约 24 GB，**装不进 15.9 GB 显存**，必须 offload。

`ai_draw.py::_load_torch_pipe()` 的策略：

| `FLUX_OFFLOAD` | 行为 | 本机（RTX 5060 Ti 16G，torch 2.9.1+cu128 / Windows） |
|---|---|---|
| `auto`（默认） | 显存 ≥24 GB 时全量驻留，否则 `enable_sequential_cpu_offload()` | 走 sequential |
| `sequential` | 逐层换入显存 | **可用**：1024×1024 ≈ 25 分钟，512×512 ≈ 6 分钟 |
| `model` | `enable_model_cpu_offload()`，整套子模型换入 | **崩**：推理第一步 `0xC0000005` 访问冲突 |
| `none` | `pipe.to(cuda)` 全量驻留 | 16 GB 装不下，会 OOM |

即：这台机器上只有 `sequential` 能真正出图。`enable_model_cpu_offload()` 在
Windows + accelerate 1.15 + torch 2.9.x 组合下会直接让进程挂掉，
所以默认不再走它（代码里保留开关，方便你在别的卡/别的 torch 版本上试）。

提速办法（按性价比）：`DRAW_SIZE=512`（4 倍速）→ 换 fp8 / GGUF 量化权重（能全量驻留，最快）→
调低 `DRAW_STEPS`（schnell 本来 4 步，不建议再降）。

冒烟脚本 `_tools/probe_flux.py` 可以按模式逐个复验（每种模式跑在独立进程，崩了不影响其它）：

```powershell
$env:PROBE_SIZE="512"
D:\AI_T_MCP\.venv312\Scripts\python.exe D:\AI_T_MCP\_tools\probe_flux.py bf16_seq
```

### 12.7 依赖版本的隐形地雷（本机踩过，务必按顺序锁）

`optimum-intel` / `diffusers` / `transformers` / `huggingface_hub` 四个包的版本约束会互相锁死：

| 组合 | 结果 |
|---|---|
| `diffusers 0.42.1`（要 hub≥1.32） + `optimum-intel 2.2.0`（要 hub<1.22） | 无解，`from huggingface_hub import resolve_revision` 直接 ImportError |
| `transformers 5.5.4`（要 hub 1.5–2.0） + `optimum-intel 2.2.0`（要 hub<1.22） | 无解 |
| **实际可用**：hub 0.36.2 + transformers 4.57.6 + diffusers 0.39.0 + optimum-intel 2.2.0 | 全链路 import 通过 |
| CPU 版 torchvision ↔ CUDA 版 torch | `operator torchvision::nms does not exist` → CLIPImageProcessor 导入失败 |

所以两条硬规则：

1. **torch 和 torchvision 必须同版本同 CUDA 变体**（cu128 配 `torchvision==0.24.1+cu128`），
   否则 `torchvision::nms` 缺失，`transformers` 一导入 CLIP 组件就崩；
2. T5 tokenizer 需要 `sentencepiece`，不装会报
   `Cannot instantiate this tokenizer from a slow version`。

## 13. AI 生产管线（克隆配音 / 位移旋转动作 / 参考视频重绘 / 素材库 / 剧本模型 / 客户端）

第 1–12 节做的是"图片 + 系统配音 + 硬编成片"。这一节把整条链路换成本地 AI 生产：
声音克隆替掉 SAPI、人物真的会动、能按参考视频生成相似视频、出图和出音都会先问素材库、
剧本由本地 4B 模型写，最后用 LobeHub 当对话客户端调 MCP。

### 13.1 六项能力与对应模块

| # | 能力 | 模块 | 缺省行为 |
|---|---|---|---|
| 1 | 声音克隆替掉 Windows SAPI | `theater/tts_clone.py`（F5-TTS 1.1.22） | 有参考音 → clone；没有参考音或没配套文字 → 自动回落 SAPI |
| 2 | 人物能动 | `theater/rigging.py` + `theater/animator.py` | 动作只有三种自由度：左右平移 / 上下平移 / 旋转；`pose_at(action, t)` 纯函数，整图 AFFINE 一次成型 |
| 3 | 按参考视频生成相似视频 | `theater/video_ref.py`（SD1.5 img2img） | 逐帧算"这帧相对上帧的偏移量"→ AI 归纳成画面描述 → 按偏移推 init image 再重绘；保留原音轨并出相似度报告 |
| 4 | 音效库 + 素材库 | `theater/library.py` | 9 个内置音效现场合成；出图/出音前先查库，命中就复用 |
| 5 | 剧本生成模型 | `pipeline_full.py::generate_script` | `LLM_BACKEND=auto`：torch 优先，OpenVINO 可选（NPU→GPU→CPU） |
| 6 | 对话客户端 | LobeHub 2.3.2 + `server.py`（13 个 MCP 工具） | 没装客户端时 MCP 仍可被任意 MCP 客户端调用 |

管线步骤（`pipeline_full.STEPS`）：`parse → assets → tts → sfx → anim → frames → video`

#### 13.1.1 人物动作只有平移和旋转

关节级肢体动画（把立绘按 alpha 连通域切成头/躯干/四肢，再绕关节旋转）在本项目里被
**主动放弃**：切分依赖抠图质量，立绘边缘毛一点就会把胳膊切成两半，旋转时露出裂缝。
改成整图仿射后，任何立绘都能动，也不挑素材。

| 动作名 | 自由度 | 幅度 / 周期 |
|---|---|---|
| `pan_left` / `pan_right` | 左右平移 | ±0.18 图宽，周期 4.0 s |
| `pan_up` / `pan_down` | 上下平移 | ±0.16 图高，周期 4.0 s |
| `rotate` | 往返旋转 | ±10°，周期 3.0 s |
| `spin` | 同向旋转 | 360°，周期 6.0 s |
| `idle` / `nod` / `wave` / `walk` / `talk` | 小幅度平移或旋转的别名 | 幅度 1.2%~9%、或 ±1.6°，供老脚本与 `build_anims` 继续使用 |

`pose_at` 返回归一化 `{dx, dy, rot}`（dx/dy 是占图片宽/高的比例），`transform_px` 换算成像素，
`warp()` 用一次 `Image.AFFINE`（绕中心旋转 + 平移，BICUBIC，透明填充）出帧，避免
"先 rotate 再 paste"造成的取整抖动与边缘拉丝。`render_clip` 的 manifest 里记
`motion_tracks`（每帧 `{t, pose, px}`），同时保留 `joint_tracks` 键作为兼容别名。

#### 13.1.2 相似视频必须"逐帧算偏移 → AI 分析 → 再生成"

`generate_similar_video` 的实际顺序（缺一步都算没做到用户要的事）：

1. **抽帧**：按 `SIM_MAX_FRAMES` 预算决定步长，得到待重绘帧序列。
2. **逐帧算偏移**：`frame_offset(prev, cur)` 缩到 `SIM_MOTION_SIZE` 灰度 → 去均值 + 汉宁窗 →
   `rfft2` 相位相关出整数位移 → 因不同 FFT 库符号约定会翻，把 `(dx,dy)` 与 `(-dx,-dy)` 都算
   一遍重叠区归一化相关系数取更像的 → 再在 ±8°、步长 2 内搜旋转并在最优点 ±1° 精修。
   相关系数直接当 `conf`，切镜/闪白这类帧对会被打低分。位移按分析尺度换算回原片像素。
3. **AI 分析**：`describe_motion` 把偏移序列归纳成人话（静止 / 向右平移 / 向下平移 + 顺时针旋转…）
   和一句英文生成提示；`SIM_MOTION_LLM=1` 时再交给本地 Qwen3-4B 改写这句提示
   （默认关：加载 7.5 GB 权重约 147 s，只在连着跑多段视频时才值）。
4. **再生成**：`prompt_all = "<用户提示>, <运动提示>"`，逐帧 img2img；
   `i>0` 且 `SIM_MOTION_WARP=1` 时，init image 用**上一帧的生成图**按本帧偏移推过去
   （`_warp_prev`），让画风与运动轨迹一路传下去，而不是每帧都从原片重新起手。
   只有"整帧都在动"才推：`frame_offset` 顺带算 `changed`（像素变化比例），
   ≥60% 相邻帧几乎不动但仍有 2% 以上像素在变 → 判为 `kind=local`（主体自己在动），
   此时推整帧会把背景一起带走，所以改回用原帧当 init。
5. **落盘可核对**：`workdir\motion.json` 存 summary / llm_hint / prompt_used / 全部 offsets；
   `out_info["motion"]` 带 summary、前 16 条 offsets、`kind` 与 `warped_frames`（真正推了几帧）。

`similarity_report()` 另外给运动一致性：两段视频各抽 6 帧做同样的逐帧分析，
比较方向点积（`direction_match`）、速度比（`speed_ratio`）与各自累计旋转；
两边都近乎静止时也算一致。这段包在 try 里，失败只写 `report["motion"]["error"]`。


### 13.2 安装（照抄，全部 D 盘）

#### 13.2.1 声音克隆（F5-TTS）

权重（ckpt 1.29 GB + 声码器 55 MB）：

```powershell
$py = "D:\AI_T_MCP\.venv312\Scripts\python.exe"
& $py D:\AI_T_MCP\_tools\mdl.py "https://modelscope.cn/models/AI-ModelScope/F5-TTS/resolve/master" "D:\ai_models\f5-tts" "D:\AI_T_MCP\_tools\mf_f5.txt" 20
curl.exe -L -C - -o D:\ai_models\vocos-mel-24khz\pytorch_model.bin "https://hf-mirror.com/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin"
curl.exe -L -C - -o D:\ai_models\vocos-mel-24khz\config.yaml   "https://hf-mirror.com/charactr/vocos-mel-24khz/resolve/main/config.yaml"
```

依赖装法见 `requirements.txt` 里「AI 声音克隆」一段：核心是一律 `--no-deps`，
且 `cached_path` 不装、改用 `_tools/cached_path_shim.py` 覆盖到 site-packages。

参考语音放 `cache\voices\`，**每段 wav 建议配同名 .txt** 写这段音频说的原话：

```
cache\voices\zhou_ref.wav   # 3~10 秒、单人、无背景音
cache\voices\zhou_ref.txt   # 这段音频对应的文字
```

没有 .txt 时程序会尝试 whisper 自动转写并缓存；离线环境转写必然失败，会打一行 warn 后回落 SAPI。

#### 13.2.2 人物动作（平移 / 旋转）

无额外依赖（Pillow + numpy 已在基线里）。立绘用透明背景、正面全身最好看，但整图仿射不挑素材，
`rig.json` 会落在 `cache\rigs\`，同一张图两次 build 签名一致。

直接调：

```powershell
& $py -c "from theater import animator; print(sorted(animator.ACTIONS)); print(animator.pose_at('pan_right', 1.0))"
# 或走 MCP：theater_animate_character(action="pan_right", seconds=2)
```

#### 13.2.3 参考视频重绘（SD1.5）

```powershell
& $py D:\AI_T_MCP\_tools\mdl.py "https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/resolve/master" "D:\ai_models\sd15-base" "D:\AI_T_MCP\_tools\mf_sd15.txt" 16
# 必须补这个小文件，否则 diffusers 报 "Can't load image processor for ..."
New-Item -ItemType Directory -Force -Path D:\ai_models\sd15-base\feature_extractor | Out-Null
curl.exe -L -o D:\ai_models\sd15-base\feature_extractor\preprocessor_config.json "https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/resolve/master/feature_extractor/preprocessor_config.json"
```

#### 13.2.4 剧本模型（Qwen3-4B）

```powershell
& $py D:\AI_T_MCP\_tools\mdl.py "https://modelscope.cn/models/Qwen/Qwen3-4B/resolve/master" "D:\ai_models\qwen3-4b" "D:\AI_T_MCP\_tools\mf_qwen.txt" 20
```

torch 格式三个 shard，bf16 约 8 GB，16 GB 显存可全量驻留（本机实测加载 147 s）。
有 Intel NPU 时设 `LLM_BACKEND=openvino` 并准备 INT4 IR；没有就留 `auto`。

#### 13.2.5 客户端（LobeHub）

```powershell
# api.github.com 可直连；资产在 github.com 上只有 40 KB/s，走镜像 + 分段并行
$rel = Invoke-RestMethod "https://api.github.com/repos/lobehub/lobehub/releases/latest" -Headers @{"User-Agent"="PowerShell"}
$exe = ($rel.assets | Where-Object { $_.name -like "*-setup.exe" }).browser_download_url
& $py D:\AI_T_MCP\_tools\pdl2.py "https://gh-proxy.com/$exe" "D:\downloads\$(Split-Path $exe -Leaf)" "<sha256>" 4
# 静默装到 D 盘：/D= 必须放最后，且不能加引号
Start-Process "D:\downloads\LobeHub-2.3.2-setup.exe" -ArgumentList "/S","/D=D:\LobeHub" -Wait
```

### 13.3 新增环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `TTS_ENGINE` | `auto` | `auto/clone/sapi`；auto = F5 权重齐就用克隆 |
| `TTS_REF_DIR` | `cache/voices` | 参考音频目录 |
| `TTS_VOICE_MAP` | 空 | `角色=参考音频文件名`（克隆引擎用） |
| `SAPI_VOICE_MAP` | 空 | `角色=SAPI 发音人名`（兜底用，与上面分开，避免撞车） |
| `TTS_SPEED` | `1.0` | 克隆语速 |
| `TTS_DEVICE` | `auto` | `auto/cuda/cpu` |
| `F5_MODEL_DIR` | `MODEL_ROOT/f5-tts/F5TTS_v1_Base` | ckpt + vocab.txt |
| `VOCOS_DIR` | `MODEL_ROOT/vocos-mel-24khz` | 声码器目录（config.yaml + pytorch_model.bin） |
| `ANIM_FPS` | `24` | 动画渲染帧率（成片仍按 `VIDEO_FPS` 编码） |
| `SIM_VIDEO_BACKEND` | `auto` | `auto/img2img/none` |
| `SIM_SD_MODEL` | `MODEL_ROOT/sd15-base` | SD1.5 目录 |
| `SIM_SIZE` / `SIM_STEPS` / `SIM_STRENGTH` | `512` / `8` / `0.45` | 重绘长边 / 采样步数 / 改动强度 |
| `SIM_MAX_FRAMES` | `240` | 最多重绘多少帧（超出按等间隔抽帧，再复制回原帧率保住时长） |
| `SIM_MOTION_SIZE` | `256` | 逐帧偏移分析的分析尺度（长边像素）；越小越快、越抗噪 |
| `SIM_MOTION_LLM` | `0` | `1` = 让本地 Qwen3-4B 把运动统计改写成一句画面描述；`0` = 用规则模板 |
| `SIM_MOTION_WARP` | `1` | `1` = 上一帧生成图按本帧偏移推成下一帧 init；`0` = 每帧都用原片 |
| `THEATER_LIBRARY` | `项目根/library` | 素材库根目录（索引 `index.json`） |
| `LIBRARY_MIN_SCORE` / `LIBRARY_FIRST` | `0.45` / `1` | 命中阈值 / 是否先查库 |
| `LLM_BACKEND` | `auto` | `auto/torch/openvino` |
| `LLM_TORCH_DIR` / `LLM_OV_DIR` | `MODEL_ROOT/qwen3-4b` / `...-int4-ov` | 两套权重 |
| `LLM_DEVICE` | `auto` | OpenVINO 设备：NPU→GPU→CPU 自动挑真实存在的 |

### 13.4 本机实测（RTX 5060 Ti 16G / Ryzen 5 9600X / 全 D 盘）

| 项目 | 实测数字 |
|---|---|
| F5-TTS 克隆单句（含首次加载模型） | 22.3 s，16 字 |
| F5-TTS 批量 2 句（模型已常驻） | 12.4 s，24 kHz mono pcm_s16le，engine=clone |
| 骨骼绑定（512×512 立绘） | 签名 `4a096c50a50e`，同图两次 build 一致（切分仍保留，用于部件级取景） |
| 位移/旋转动作 | `pan_left` 在 1.0 s 时 dx=-0.18（1000 宽即 -180 px）；`rotate` 在 0.75 s 时 >9°；`spin` 5.9 s 时仍在 0~360° |
| 逐帧偏移分析（已知位移校准） | 期望 (60,20) → 实测 (60,20) conf 0.918；期望 (-45,0) → (-45,0) conf 0.976；期望 (0,-30) → (0,-30) conf 0.946；期望 rot=6° → 6.0° conf 0.998 |
| 动画渲染 | 2 镜头共 144 帧（60 + 84），24 fps，27.0 s（含逐帧合成字幕与背景） |
| 内置音效 | 9 个现场合成入库，0.7 MB，耗时 < 1 s |
| 成片（2 镜头：克隆配音 + 音效 + 动画） | 6.02 s / 1920×1080 / 30 fps / 180 帧 / 1.74 MB，NVENC 编码 5.2 s |
| 动作真实性核对（旋转 `rotate`） | 抽 0.2 s 与 1.0 s 两帧：mean_abs_diff 0.0085，3.6% 像素在动 |
| 动作真实性核对（平移 `pan_right`） | 同法：mean_abs_diff 0.0166，6.8% 像素在动（人物 0.2~1.0 s 内右移约 240 px） |
| 参考视频 → 相似视频（24 帧，旧版对照） | 24.2 s（当时未做逐帧运动分析），出片 6.0 s / 30 fps / 1920×1080 / 180 帧 / 保留了原音轨 |
| 参考视频 → 相似视频（8 帧，含逐帧运动分析） | 226.4 s 里首帧 209.9 s（含 SD 管道加载 82 s 与 CUDA 预热）、其余 0.7~1.1 s/帧；出片 3.0 s / 24 fps / 1920×1080 / 保留原音轨 |
| 相似度报告 | 时长偏差 0.0%，fps 与分辨率一致，边缘 IoU 均值 0.34 |
| 逐帧运动分析 → 全画面平移参考（8 帧） | 判为 `global` 向左平移 84 px/帧；生成片实测 82.5 px/帧，`direction_match=true`，`speed_ratio=1.027`；7/7 帧用上一帧生成图按偏移推 init |
| 逐帧运动分析 → 局部动作参考（人物在静态背景里平移） | 判为 `local`（3% 像素在变），不推整帧，改回用原帧当 init |
| 逐帧运动分析 → 自家动画成片 | 判为 `static`（0.3% 像素在变），阈值不误报 |
| SD1.5 img2img 稳态单帧 | 512 px 重绘再放大回 1080p：首帧含 CUDA 预热 ~30 s，之后 0.7~1.1 s/帧（直接在 1080p 上跑是 75 s/帧） |
| 剧本生成（Qwen3-4B bf16） | 加载 147 s + 生成 6 镜头 32 s（合计 179.8 s） |
| FLUX 出图（基线，供对比） | 512 px ≈ 21 分钟 / 1024 px 1504 s（sequential offload） |

### 13.5 这一节踩过的坑（症状 → 原因 → 解法）

| 症状 | 原因 | 解法 |
|---|---|---|
| `use_ema=False` 报一大串 Missing/Unexpected key | 官方 ckpt 里键名全是 `ema_model.*` | `F5TTS(..., use_ema=True)` |
| 克隆时仍联网要下 vocos | F5 自己走 `hf_hub_download`，monkeypatch `Vocos.from_pretrained` 无效 | 传 `vocoder_local_path=<同时有 config.yaml 与 pytorch_model.bin 的目录>` |
| `OSError: [WinError 127]` 读音频 | torchaudio 2.9 的 `load` 改走 torchcodec，torchcodec 要共享版 ffmpeg（本项目是静态构建） | `_patch_f5_audio_io()` 把 `load` 换成 soundfile；重采样仍用 `torchaudio.functional` |
| `import datasets` 报 `BucketNotFoundError` | datasets 5.x 要 hub 新符号，本地 hub 是 0.36.2 | 钉 `datasets==3.6.0` |
| `import cached_path` 永远卡在 `google.cloud` | 0.1.0~1.8.10 都这样，补 google 包只会越补越深 | 卸掉 cached_path，用 `_tools/cached_path_shim.py` 覆盖 |
| `Option framerate not found` | `-framerate` 属 image2 demuxer，concat demuxer 不认 | 帧序列用 `-r <fps> -f concat -safe 0 -i list.txt` |
| 成片比预期短 1.3 s | `-shortest` 让镜头被较短的音轨截断（台词后面本来留白） | 去掉 `-shortest`，改 `-af apad` + `-t <时长>` |
| SD1.5 加载报 `Can't load image processor` | 只下了 unet/vae/text_encoder，缺 `feature_extractor/preprocessor_config.json` | 从 ModelScope 补这一个小文件 |
| `ModuleNotFoundError: mcp.server.fastmcp` | mcp 2.x 把 FastMCP 改名成 MCPServer | `server.py` 先试 1.x，失败再 fallback `mcp.server.mcpserver` |
| pdl.py 下完的安装包哈希不对、装不上 | 分段拼接错乱（大小对、内容错） | 换 `_tools/pdl2.py`：逐段校验长度 + 整文件 sha256，不过就不算下完 |
| 镜像下大文件 HTTP 429 | 并发太高被限流 | 降到 4 线程 + 429 长退避 + 断点续传 |
| 测到的旋转角与 `warp()` 转向相反 | PIL `rotate(角度)` 逆时针，而 `warp()` 按图像坐标（y 向下）正方向顺时针 | `_gray_rotated` 里 `im.rotate(-deg)`，让测出来的角度与 warp 同号 |
| 人物在静止背景里走，逐帧分析却报 0 px | 相位相关看整帧，大面积静态背景把峰值压回 (0,0) | 同时算 `changed` 像素变化比例，≥2% 且整帧没位移 → 判 `local`（主体在动），并按原帧当 init，不推整帧 |
| 上一帧生成图当 init 导致画风/构图逐帧漂移 | 每帧都把上帧再推一次，误差会累积 | 只在 `kind=global` 且位移 ≥2 px（或旋转 ≥0.8°）、`conf` ≥0.35 时才推；否则回用原帧 |

### 13.6 客户端接入（LobeHub + MCP）

仓库内已提供 `mcp/lobe-mcp.json`：

```json
{
  "mcpServers": {
    "local-theater-auto": {
      "command": "D:\\AI_T_MCP\\.venv312\\Scripts\\python.exe",
      "args": ["D:\\AI_T_MCP\\server.py"],
      "env": { "TTS_ENGINE": "auto", "LLM_BACKEND": "auto" }
    }
  }
}
```

在 LobeHub「设置 → MCP / 插件」里粘贴这段即可看到 13 个工具：
`theater_check_env` / `theater_parse_script` / `theater_generate_script` / `theater_generate_assets` /
`theater_generate_tts` / `theater_compose_frames` / `theater_export_video` / `theater_run_all` /
`theater_clone_voice` / `theater_animate_character` / `theater_generate_similar_video` /
`theater_library_search` / `theater_make_sfx`

典型对话：先说"用 6 个镜头写一个深夜机房的故事"，再"生成素材和配音"，
最后"导出成片"——三个工具依次串起来，中间产物全部落在 `cache\` 和 `library\`，重跑幂等。

### 13.7 复验命令

```powershell
cd D:\AI_T_MCP
$py = "D:\AI_T_MCP\.venv312\Scripts\python.exe"
& $py check_env.py                        # 期望 FAIL 0 项
& $py -m unittest discover -s tests       # 期望 71/71 OK
& $py _tools\smoke_clone.py               # 克隆一句，真出 wav
& $py _tools\e2e_demo.py                  # 2 镜头全链路 → cache\demo_animated.mp4
& $py _tools\probe_similar.py             # 参考视频 → 逐帧偏移 + 相似视频 + 相似度报告
& $py _tools\probe_motion.py              # 已知位移校准 frame_offset（四处 expect 对照）
```