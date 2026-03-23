# 基于深度学习的社交媒体情感分析系统

Social Media Sentiment Analysis System with Deep Learning

[English](README_en.md) | 中文

## 1. 项目简介

本项目是一个基于深度学习的社交媒体文本情感分析系统，支持对包含表情符号、话题标签、网络用语等噪声的短文本进行情感分类（正面/负面）。

> **注意**：本系统默认使用二分类（正面/负面），配套微博情感数据集进行训练。

### 核心特性

- **双通道融合架构**：结合预训练语言模型与社交特征提取
- **强大的文本预处理**：Emoji 转换、HTML/URL 清理、话题标签提取
- **全方位评估指标**：准确率、精确率、召回率、F1 值、混淆矩阵
- **RESTful API**：基于 FastAPI 的异步推理服务
- **现代化前端界面**：Vue 3 + Vite 交互界面

## 2. 系统架构

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
│  │ [CLS] vector       │    │ 10-dim → 128-dim         │  │
│  │ 768-dim            │    │                            │  │
│  └─────────┬──────────┘    └──────────┬─────────────────┘  │
│            │                          │                     │
│            └────────┬─────────────────┘                     │
│                     ▼                                      │
│            ┌────────────────┐                               │
│            │ Fusion &      │                               │
│            │ Classification │                               │
│            │ Concat → MLP  │                               │
│            │ → 2 logits    │                               │
│            └────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

## 3. 模型架构详解

### Text Encoder (通道 A)

使用哈工大开源的 [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) 作为文本编码器：

- **架构**：BERT-based, Whole Word Masking (wwm)
- **隐藏层维度**：768
- **功能**：提取文本的全局语义表征，取 `[CLS]` token 对应向量

### Social Feature Extractor (通道 B)

MLP 网络，用于将社交信号转为稠密向量：

```
Input (10-dim) → Linear(10, 128) → ReLU → LayerNorm(128) → Output (128-dim)
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
[CLS](768) + Social(128) = 896-dim
    → Linear(896, 128) → ReLU → Dropout(0.1) → Linear(128, 2) → logits
```

## 4. 目录结构

```
deepLeran/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # API 路由与 Pydantic 模型
│   │   ├── core/
│   │   │   └── trainer.py        # 训练循环与评估指标
│   │   ├── dataset/
│   │   │   ├── dataset.py         # PyTorch Dataset 类
│   │   │   └── text_cleaner.py    # 文本清洗工具
│   │   ├── models/
│   │   │   └── model.py           # 双通道融合模型
│   │   └── main.py                # FastAPI 入口
│   ├── checkpoints/               # 模型权重保存目录
│   ├── data/
│   │   ├── raw/                   # 原始数据
│   │   └── processed/             # 处理后数据 (train/val/test)
│   ├── scripts/
│   │   ├── prepare_data.py        # 数据预处理脚本
│   │   ├── train.py               # 模型训练脚本
│   │   ├── export_model_graph.py  # 模型计算图导出
│   │   ├── test_cleaner.py        # TextCleaner 测试
│   │   └── test_dataset.py        # Dataset 测试
│   ├── pyproject.toml             # Python 依赖
│   └── uv.lock                    # 依赖锁定文件
├── frontend/
│   ├── src/
│   │   ├── App.vue                # 主界面组件
│   │   ├── api/
│   │   │   └── sentiment.ts        # API 调用模块
│   │   ├── style.css              # 全局样式
│   │   └── main.ts                # Vue 入口
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── CLAUDE.md                      # 开发指南
└── README.md
```

## 5. 从 GitHub 克隆后的完整配置流程

### 1. 克隆项目

```bash
git clone <repository-url>
cd deepLeran
```

### 2. 配置后端环境

```bash
cd backend

# 使用 uv 安装依赖
uv sync
```

### 3. 准备数据集

本项目使用微博情感分析数据集（二分类：正面/负面）。

**方式一**：使用已有的数据集文件

将你的数据集 CSV 文件放入 `backend/data/raw/` 目录，文件格式应为：
```csv
label,review
1,这是一条正面评论
0,这是一条负面评论
```

**方式二**：如使用 weibo_senti_100k.csv，运行数据预处理：

```bash
# 确保数据文件在 backend/data/raw/weibo_senti_100k.csv
uv run python scripts/prepare_data.py
```

输出结果：
```
Loaded 119,988 rows
Train: 95,990 rows
Val:   11,999 rows
Test:  11,999 rows
Saved to backend/data/processed/
```

### 4. 训练模型

```bash
cd backend

# 训练模型（约需 15-30 分钟，视 GPU 性能）
uv run python scripts/train.py
```

**训练配置**（可在 `scripts/train.py` 中修改）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 5 | 训练轮次 |
| batch_size | 32 | 批大小 |
| learning_rate | 2e-5 | 学习率 |
| max_length | 128 | 文本最大长度 |

训练过程中会打印每个 epoch 的：
- Train Loss
- Val Accuracy
- Val F1 Score (macro/weighted)
- Confusion Matrix

训练完成后，最佳模型自动保存到 `backend/checkpoints/best_model.pt`

### 5. 启动后端服务

```bash
cd backend

# 启动 API 服务
uv run uvicorn app.main:app --reload --port 8000
```

验证服务运行：
```bash
curl http://localhost:8000/health
```

### 6. 启动前端

```bash
cd frontend

# 安装依赖（如未安装）
npm install

# 启动开发服务器
npm run dev
```

前端访问地址：http://localhost:5173

## 6. API 使用

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
    "negative": 0.147
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

## 7. 模型可视化

### 1. 导出计算图

本项目支持导出神经网络计算图用于可视化分析。

```bash
cd backend

# 安装依赖（自动安装 netron 和 onnxscript）
uv sync

# 导出模型计算图到 ONNX 格式
uv run python scripts/export_model_graph.py
```

导出后模型文件位于 `backend/model_graph.onnx`（已移动至项目根目录 `model_graph.onnx`）。

### 2. 使用 Netron 可视化

**方式一：在线查看**
1. 访问 https://netron.app
2. 点击 "Open Model" 上传 `model_graph.onnx` 文件

**方式二：本地启动 Netron 服务器**
```bash
# 在项目根目录运行
netron model_graph.onnx
```

### 3. 计算图说明

导出的 ONNX 模型包含以下输入输出：

| 名称 | 形状 | 说明 |
|------|------|------|
| input_ids | (batch_size, sequence_length) | 文本 Token IDs |
| attention_mask | (batch_size, sequence_length) | 注意力掩码 |
| social_features | (batch_size, 10) | 社交特征向量 |
| logits | (batch_size, num_labels) | 情感分类 logits |


## 8. 依赖列表

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

## 9. 开发指南

### 代码规范

- **Python**: 严格使用 Type Hints，运行 `uvx pyright app/` 确保无类型错误
- **前端**: Vue 3 Composition API + `<script setup>` 语法
- **提交规范**: 遵循 Conventional Commits (feat/fix/docs/style/refactor/test/chore)

### 运行测试

```bash
# Python 类型检查
cd backend
uvx pyright app/

# 前端构建检查
cd frontend
npm run build
```

## 10. 常见问题

### Q: 启动时报 `size mismatch` 错误？

A: 检查 `app/main.py` 中的 `ModelConfig` 参数是否与训练时一致，特别是 `fusion_hidden_dim`、`num_labels`。

### Q: HuggingFace 模型下载失败？

A: 设置 HF_TOKEN 环境变量或配置镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 训练太慢？

A: 确保使用 GPU 加速。macOS M1/M2/M3 会自动使用 MPS，NVIDIA GPU 会使用 CUDA。

### Q: 如何使用自己的数据集？

A: 确保 CSV 文件格式为 `label,review`，label 为 0（负面）或 1（正面），然后运行 `prepare_data.py`。

## 11. License

MIT License

## 12. 致谢

- 哈工大：[hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
- HuggingFace Transformers
- PyTorch Team
