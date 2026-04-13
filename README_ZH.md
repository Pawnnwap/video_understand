# 视频理解管道

一个综合视频分析系统，将讲座/录像视频转换为可查询的知识库，使用语音识别、视觉分析和语义搜索。

## 功能特性

- **语音转文字**: FunASR (paraformer-zh) 中文原生转录，带时间戳
- **视觉分析**: VLM帧分析 + PaddleOCR 幻灯片内容提取
- **语义搜索**: ChromaDB向量存储用于知识检索
- **交互式CLI**: 自然语言查询已处理视频

## 系统要求

- Python 3.10+
- ffmpeg（用于音频提取）
- LM Studio（用于VLM/LLM推理）
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
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio端点 |
| `LM_STUDIO_API_KEY` | `lm-studio` | API密钥 |
| `VLM_MODEL` | `qwen3.5-4b` | 视觉模型名称 |
| `LLM_MODEL` | `qwen3.5-4b` | 语言模型名称 |

CLI参数覆盖（最高优先级）：

```bash
python cli.py video.mp4 --base-url http://localhost:1234/v1 --vlm-model qwen3.5-4b
python pipeline.py URL --api-key your-key --llm-model qwen3.5-4b
python query.py ./video_db/project --model qwen3.5-4b
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
# 交互式REPL
python cli.py
python query.py ./video_db/my_lecture

# 单次查询
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
| `/summary` | 综合摘要 |
| `/headline` | 一行标题 |
| `/brief` | 3-5句概述 |
| `/outline` | 主题大纲 |
| `/slides` | 列出幻灯片变化 |
| `/transcript` | 完整转录 |
| `/at MM:SS [问题]` | 时间戳查询 |
| `/knowledge <主题>` | 深度提取 |
| `<任意文本>` | 语义搜索 |

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

超时参数在 `config.py` 中配置：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `VLM_CALL_TIMEOUT_S` | 120 | VLM HTTP调用超时 |
| `LLM_CALL_TIMEOUT_S` | 60 | LLM融合/查询超时 |
| `OCR_TIMEOUT_S` | 60 | OCR子进程超时 |
| `FFMPEG_TIMEOUT_S` | 300 | ffmpeg子进程超时 |
| `FUNASR_TIMEOUT_S` | 0 | FunASR超时（0=无限制） |

## 项目结构

```
video_summarize/
├── cli.py              # 交互式工作空间CLI
├── pipeline.py         # 主处理管道
├── query.py            # 独立查询界面
├── config.py           # 配置
├── downloader.py       # 视频下载(yt-dlp)
├── retry.py            # 重试工具
├── core/
│   ├── stt.py          # 语音转文字(FunASR)
│   ├── frame_sampler.py # 自适应帧提取
│   ├── vlm_analyser.py # 视觉分析
│   ├── fusion.py       # 语音-视觉融合
│   ├── database.py     # ChromaDB + 时间线
│   └ ocr_worker.py   # OCR子进程
├── query/
│   └ query_engine.py # RAG查询引擎
├── utils/
│   └ retry.py        # 重试重导出
└── video_db/           # 已处理项目存储
```

## 故障排除

**ffmpeg未找到**: 安装ffmpeg并添加到PATH。

**CUDA内存不足**: 在config.py中设置`FUNASR_DEVICE=cpu`。

**LM Studio连接失败**: 确认LM Studio运行且模型已加载。

**PaddleOCR失败**: 无CUDA时设置`OCR_USE_GPU=False`。

## 许可证

MIT
