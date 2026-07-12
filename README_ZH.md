# 视频理解管道

一个综合视频分析系统，将讲座/录像视频转换为可查询的知识库，使用语音识别、视觉分析和语义搜索。

## 功能特性

- **语音转文字**: FunASR (paraformer-zh) 中文原生转录，带时间戳
- **视觉分析**: VLM帧分析 + RapidOCR (ONNX) 幻灯片内容提取
- **语义搜索**: ChromaDB向量存储用于知识检索
- **智能体网络核查**: 通过 OpenCode 智能体检索并读取网络来源，对视频声明进行事实核查
- **交互式CLI**: 自然语言查询已处理视频

## 系统要求

- Python 3.10+
- ffmpeg（用于音频提取）
- [opencode](https://opencode.ai) CLI（用于VLM/LLM推理，内置免费模型）
- 推荐CUDA显卡

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd video_summarize

# 创建conda环境
conda create -n video python=3.10
conda activate video

# 安装依赖
pip install -r requirements.txt
```

## 配置

配置通过 `config.py` 或环境变量：

| 环境变量 | 默认值 | 描述 |
|----------|--------|------|
| `VLM_MODEL` | `opencode/mimo-v2.5-free` | 视觉模型（opencode提供商） |
| `VLM_VARIANT` | `low` | VLM 推理变体（low/medium/high）—— 帧分析保持最低思考 |
| `OPENCODE_SERVER_PORT` | `0`（随机） | opencode服务器端口 |
| `LLM_MODEL` | 同VLM_MODEL | 用于融合、查询和核查的 OpenCode 文本模型 |
| `LLM_VARIANT` | `high` | 文本模型推理变体（low/medium/high）—— LLM 工作默认最高思考 |

### FunASR（语音识别）

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `FUNASR_MODEL` | `paraformer-zh` | ASR模型 |
| `FUNASR_VAD_MODEL` | `fsmn-vad` | 语音活动检测模型 |
| `FUNASR_PUNC_MODEL` | `ct-punc` | 标点模型 |
| `FUNASR_DEVICE` | `cuda` | 设备（cuda/cpu） |
| `FUNASR_LANGUAGE` | `zh` | 语言代码 |
| `FUNASR_TIMEOUT_S` | `0` | 超时（0=无限制） |
| `STT_SENTENCE_SPLIT_GAP_MS` | `500` | 句子分割间隙阈值（毫秒） |

### OCR（RapidOCR, ONNX）

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `OCR_MODEL_NAME` | `PP-OCRv5_mobile` | OCR模型名称（仅供参考） |
| `OCR_LANG` | `ch` | OCR语言 |
| `OCR_USE_GPU` | `True` | OCR使用GPU（onnxruntime CUDA） |
| `OCR_MIN_CONFIDENCE` | `0.6` | 最小置信度阈值 |
| `OCR_TIMEOUT_S` | `60` | OCR子进程超时 |
| `OCR_RICH_TEXT_MIN_LINES` | `3` | 富文本检测最小行数 |

### 帧采样

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `SENTENCE_END_OFFSET_MS` | `200` | 句子结束前偏移（毫秒） |
| `LONG_PAUSE_THRESHOLD_MS` | `800` | 长暂停检测阈值（毫秒） |
| `FALLBACK_FPS_FLOOR` | `0.2` | 静默段后备帧率 |
| `FRAME_MAX_DIM` | `768` | 最大帧尺寸（像素） |
| `FRAME_QUALITY` | `75` | JPEG质量 |
| `FRAME_SIMILARITY_THRESHOLD` | `0.90` | 帧相似度阈值 |

### VLM / LLM

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `VLM_MAX_TOKENS` | `512` | VLM最大输出tokens |
| `VLM_TEMPERATURE` | `0.1` | VLM温度 |
| `VLM_CALL_TIMEOUT_S` | `120` | VLM HTTP调用超时 |

### 重试配置

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `RETRY_MAX_ATTEMPTS` | `4` | 最大重试次数 |
| `RETRY_BASE_DELAY_S` | `2.0` | 基础重试延迟（秒） |
| `RETRY_MAX_DELAY_S` | `30.0` | 最大重试延迟（秒） |
| `RETRY_JITTER_FACTOR` | `0.25` | 重试抖动因子 |

### 数据库与嵌入

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `DB_DIR` | `./video_db` | 数据库存储目录 |
| `CHROMA_COLLECTION` | `segments` | ChromaDB集合名称 |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 嵌入模型 |
| `FUSION_SEGMENT_SIZE` | `5` | 每批融合段数 |

### 下载与FFmpeg

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `DOWNLOAD_MAX_DURATION_SEC` | `0` | 最大下载时长（0=无限制） |
| `FFMPEG_TIMEOUT_S` | `300` | ffmpeg子进程超时 |
| `FFMPEG_EXTRACTION_TIMEOUT_S` | `60` | 帧提取超时 |
CLI参数覆盖（最高优先级）：

```bash
python cli.py video.mp4 --vlm-model opencode/mimo-v2.5-free --text-model opencode/mimo-v2.5-free
python pipeline.py URL --text-model opencode/mimo-v2.5-free
python query.py ./video_db/project --text-model opencode/mimo-v2.5-free --ask "主要话题是什么？"
```

## 使用方法

### 处理视频

```bash
# 本地文件
python cli.py lecture.mp4

# YouTube/Bilibili链接
python cli.py https://www.youtube.com/watch?v=...
python cli.py BV1GE411T7Wv
python cli.py dQw4w9WgXcQ  # YouTube ID

# 直接管道
python pipeline.py video.mp4 --force  # 强制重新处理
```

### 查询已处理视频

```bash
# 交互式项目会话
python cli.py
python cli.py ./video_db/my_lecture

# 单次查询包装器
python query.py ./video_db/my_lecture --ask "主要话题是什么？"
python query.py ./video_db/my_lecture --summary
python query.py ./video_db/my_lecture --at 05:30 --question "显示什么幻灯片？"
```

### CLI命令

交互式工作空间内：

| 命令 | 描述 |
|------|------|
| `list` / `ls` | 列出已处理项目 |
| `open <名称|#>` | 进入项目 |
| `process <路径>` | 处理新视频 |
| `<BV/URL>` | 下载、处理并打开 |

项目内：

| 命令 | 描述 |
|------|------|
| `/summary [style]` | 全视频摘要：comprehensive（默认）、brief、headline |
| `/outline` | 主题大纲 |
| `/slides` | 列出幻灯片变化 |
| `/transcript` | 完整转录 |
| `/at MM:SS [问题]` | 时间戳查询 |
| `/crosscheck [n]` | 使用 OpenCode 智能体进行网络事实核查（默认5条） |
| `<任意问题>` | 自然语言提问（推荐）—— 具体问题直接回答，宽泛主题自动深度解读 |

## 架构

```
video.mp4
    │
    ▼ [阶段1: 语音识别]
transcript.json  (带时间戳的句子片段)
    │
    ▼ [阶段2a: 帧采样]
frame_schedule.json  (自适应触发器)
    │
    ▼ [阶段2b: 视觉分析]
frame_analyses.json  (每帧OCR + VLM)
    │
    ▼ [阶段3: 融合]
fused_segments.json  (语音+视觉合并)
    │
    ▼ [阶段4: 数据库]
chroma/  (向量存储)
timeline.json  (结构化时间线)
```

## 超时配置

超时参数在 `config.py` 中配置（见上方配置部分）。
## 项目结构

```
video_summarize/
├── cli.py                    # 交互式工作空间CLI
├── pipeline.py               # 主处理管道
├── query.py                  # 独立查询界面
├── config.py                 # 配置（所有可调参数）
├── downloader.py             # 视频下载（yt-dlp）
├── requirements.txt          # 依赖
├── core/
│   ├── lang.py               # 语言检测
│   ├── stt.py                # 语音转文字（FunASR）
│   ├── fusion.py             # 语音-视觉融合
│   ├── database.py           # ChromaDB + 时间线
│   └── vision/
│       ├── __init__.py       # 视觉模块初始化
│       ├── frame_sampler.py  # 自适应帧提取
│       ├── vlm_analyser.py   # VLM帧分析 + 内联RapidOCR
│       └── opencode_vlm.py   # opencode serve 子进程 + HTTP 客户端
├── query/
│   ├── __init__.py           # 查询模块初始化
│   ├── query_engine.py       # RAG查询引擎
│   └── crosscheck.py         # 网络事实核查管道
├── utils/
│   ├── __init__.py           # 工具模块初始化
│   ├── video.py              # 视频工具
│   ├── retry.py              # 重试工具
│   └── logging_setup.py      # 日志配置
├── models/                   # 内置离线模型
└── video_db/                 # 已处理项目存储
```

## 故障排除

**ffmpeg未找到**: 安装ffmpeg并添加到PATH。

**CUDA内存不足**: 在config.py中设置`FUNASR_DEVICE=cpu`。

**LM Studio连接失败**: 确认LM Studio运行且模型已加载。

**RapidOCR失败**: 无CUDA或onnxruntime-gpu问题时设置`OCR_USE_GPU=False`。

## 许可证

MIT
