# 基于深度学习的社交媒体情感分析系统

Social Media Sentiment Analysis System with Deep Learning

[English](README_en.md) | 中文

## 项目简介

本项目是一个基于深度学习的社交媒体文本情感分析系统，支持对包含表情符号、话题标签、网络用语等噪声的短文本进行情感分类（正面/负面/中性）。

### 核心特性

- **双通道融合架构**：结合预训练语言模型与社交特征提取
- **强大的文本预处理**：Emoji 转换、HTML/URL 清理、话题标签提取
- **全方位评估指标**：准确率、精确率、召回率、F1 值、混淆矩阵
- **RESTful API**：基于 FastAPI 的异步推理服务
- **现代化前端界面**：Vue 3 + Vite 交互界面

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Vue.js Frontend                          │
│                 (http://localhost:5173)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP JSON
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend                             │
│                 (http://localhost:8000)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   CORS      │  │   Routes    │  │  Model Lifecycle    │ │
│  │  Middleware │  │ /api/predict│  │  (Lifespan)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              SocialSentimentFusionModel                     │
│  ┌────────────────────┐    ┌────────────────────────────┐  │
│  │ Channel A           │    │ Channel B                  │  │
│  │ Text Encoder        │    │ Social Feature Extractor   │  │
│  │ (RoBERTa-wwm-ext)  │    │ (MLP: Linear→ReLU→LayerNorm) │
│  │ [CLS] vector       │    │ 10-dim → 64-dim          │  │
│  │ 768-dim            │    │                            │  │
│  └─────────┬──────────┘    └──────────┬─────────────────┘  │
│            │                          │                     │
│            └────────┬─────────────────┘                     │
│                     ▼                                      │
│            ┌────────────────┐                               │
│            │ Fusion &      │                               │
│            │ Classification │                               │
│            │ Concat → MLP  │                               │
│            │ → 3 logits    │                               │
│            └────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

## 模型架构详解

### Text Encoder (通道 A)

使用哈工大开源的 [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) 作为文本编码器：

- **架构**：BERT-based, Whole Word Masking (wwm)
- **隐藏层维度**：768
- **功能**：提取文本的全局语义表征，取 `[CLS]` token 对应向量

### Social Feature Extractor (通道 B)

MLP 网络，用于将社交信号转为稠密向量：

```
Input (10-dim) → Linear(10, 64) → ReLU → LayerNorm(64) → Output (64-dim)
```

输入特征包括：
| 索引 | 特征 | 说明 |
|------|------|------|
| 0 | emoji_count | Emoji 数量 |
| 1 | hashtag_count | 话题标签数量 |
| 2 | mention_count | @提及数量 |
| 3 | exclamation_count | 感叹号数量 |
| 4 | question_count | 问号数量 |
| 5 | url_count | URL 数量 |
| 6 | html_count | HTML 标签数量 |
| 7 | text_length_norm | 文本长度（归一化） |
| 8 | emoji_polarity_sum | Emoji 极性求和 |
| 9 | punctuation_intensity | 标点符号强度 |

### Fusion & Classification Head

```
[CLS](768) + Social(64) = 832-dim
    → Linear(832, 128) → ReLU → Dropout(0.1) → Linear(128, 3) → logits
```

## 目录结构

```
deepLeran/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # API 路由与 Pydantic 模型
│   │   ├── core/
│   │   │   └── trainer.py          # 训练循环与评估指标
│   │   ├── dataset/
│   │   │   ├── dataset.py          # PyTorch Dataset 类
│   │   │   └── text_cleaner.py     # 文本清洗工具
│   │   ├── models/
│   │   │   └── model.py            # 双通道融合模型
│   │   └── main.py                 # FastAPI 入口
│   ├── checkpoints/                 # 模型权重保存目录
│   ├── data/
│   │   ├── raw/                     # 原始数据
│   │   └── processed/               # 处理后数据
│   ├── scripts/
│   │   ├── test_cleaner.py          # TextCleaner 测试
│   │   └── test_dataset.py           # Dataset 测试
│   ├── pyproject.toml               # Python 依赖
│   └── uv.lock                      # 依赖锁定文件
├── frontend/
│   ├── src/
│   │   ├── App.vue                  # 主界面组件
│   │   ├── api/
│   │   │   └── sentiment.ts          # API 调用模块
│   │   ├── style.css                # 全局样式
│   │   └── main.ts                  # Vue 入口
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── CLAUDE.md                        # 开发指南
└── README.md
```

## 环境配置

### 前置要求

- Python 3.10+
- Node.js 18+
- npm 或 yarn
- uv (Python 包管理器)

### 后端配置

```bash
# 进入后端目录
cd backend

# 使用 uv 同步依赖
uv sync

# 或手动安装
uv add fastapi uvicorn torch transformers datasets scikit-learn pandas numpy tqdm rich pydantic python-multipart
```

### 前端配置

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install
```

## 运行项目

### 1. 启动后端

```bash
cd backend

# 开发模式（热重载）
uv run uvicorn app.main:app --reload --port 8000

# 或直接运行
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

后端启动后：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 2. 启动前端

```bash
cd frontend

# 开发模式
npm run dev

# 构建生产版本
npm run build
```

前端启动后访问：http://localhost:5173

### 3. 训练模型（如有数据集）

```bash
cd backend

# 查看训练脚本用法
uv run python scripts/train.py --help

# 基本训练命令
uv run python scripts/train.py \
    --data_path data/processed/train.csv \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5
```

## API 使用

### 分析文本情感

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "今天心情太好了！😊 #开心"}'
```

**响应示例**：

```json
{
  "sentiment": "positive",
  "confidence": 0.853,
  "confidence_per_class": {
    "positive": 0.853,
    "negative": 0.102,
    "neutral": 0.045
  },
  "social_features": {
    "emoji_count": 1,
    "emoji_descriptions": ["[微笑]"],
    "hashtag_count": 1,
    "hashtags": ["#开心"],
    "mention_count": 0,
    "mentions": [],
    "url_count": 0,
    "exclamation_count": 1,
    "question_count": 0,
    "cleaned_text": "今天心情太好了！ [微笑] #开心"
  },
  "raw_text": "今天心情太好了！😊 #开心"
}
```

### 获取可用标签

```bash
curl http://localhost:8000/api/labels
```

## 依赖列表

### 后端

| 包 | 版本 | 说明 |
|------|------|------|
| fastapi | ≥0.115 | Web 框架 |
| uvicorn | ≥0.34 | ASGI 服务器 |
| torch | ≥2.5 | 深度学习框架 |
| transformers | ≥4.48 | 预训练模型 |
| datasets | ≥3.2 | 数据集处理 |
| scikit-learn | ≥1.6 | 评估指标 |
| pandas | ≥2.2 | 数据处理 |
| numpy | ≥2.2 | 数值计算 |
| tqdm | ≥4.67 | 进度条 |
| rich | ≥13.9 | 命令行美化 |
| pydantic | ≥2.10 | 数据校验 |

### 前端

| 包 | 版本 | 说明 |
|------|------|------|
| vue | ^3.5 | 渐进式框架 |
| vite | ^8.0 | 构建工具 |
| pinia | ^3.0 | 状态管理 |
| vue-router | ^4.6 | 路由管理 |

## 开发指南

### 代码规范

- **Python**: 严格使用 Type Hints，运行 `uvx pyright` 确保无类型错误
- **前端**: Vue 3 Composition API + `<script setup>` 语法
- **提交规范**: 遵循 Conventional Commits (feat/fix/docs/style/refactor/test/chore)

### 运行测试

```bash
# Python 类型检查
cd backend
uvx pyright app/

# Python 单元测试
uv run pytest tests/

# 前端类型检查
cd frontend
npm run build
```

## 常见问题

### Q: 启动时报 `size mismatch` 错误？

A: 检查 `app/main.py` 中的 `ModelConfig` 参数是否与训练时一致，特别是 `fusion_hidden_dim`。

### Q: HuggingFace 模型下载失败？

A: 设置 HF_TOKEN 环境变量或配置镜像源。

### Q: macOS 上 PyTorch 使用 MPS 加速？

A: 模型会自动检测并使用 MPS（需要 PyTorch 2.0+）。

## License

MIT License

## 致谢

- 哈工大：[hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
- HuggingFace Transformers
- PyTorch Team
