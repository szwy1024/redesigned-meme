# Social Media Sentiment Analysis System - 毕业设计开发指南

## 🎯 项目概述
本项目是一个基于深度学习的社交媒体文本情感分析系统。系统需要处理包含表情符号、话题标签、网络用语等噪声的非结构化短文本，将其分类为正面、负面或中性。
开发流程涵盖：数据爬取/清洗 -> 多模型对比实验（传统ML、经典DL、预训练模型） -> 特征融合研究 -> Web 交互系统部署。

## 🛠️ 技术栈与工具链
- **包管理与虚拟环境**: `uv` (严格使用 uv 进行 Python 依赖管理)
- **静态类型检查**: `uvx pyright`
- **版本控制**: `git` (保持高频、有意义的 commit)
- **后端算法与 API**: Python, PyTorch, HuggingFace Transformers, FastAPI
- **前端界面**: Vue.js + Vite (使用组合式 API 和单文件组件 `.vue`)

## 🧠 推荐模型架构基线 (可灵活调整)
为了满足任务书中“预训练模型微调”与“特征融合”的要求，初始核心模型采用以下混合架构：

1.  **文本特征提取 (Text Encoder)**: 
    - 使用 `RoBERTa-wwm-ext` (中文) 作为基线的主干网络提取文本深层语义特征的 `[CLS]` 向量。
    - *后期调整空间*：为了完成多模型对比，这里可以轻松替换为 `TextCNN` 或 `BiLSTM` 提取的特征向量。
2.  **社交特征提取 (Social Feature Extractor)**:
    - 独立模块：通过词典或规则提取文本中的表情符号极性、标点符号强度（如多个感叹号）、网络用语标识，并映射为固定维度的 Dense Vector (稠密向量)。
3.  **特征融合与分类层 (Fusion & Classification Head)**:
    - 将 Text Encoder 输出的 $d_1$ 维向量与 Social Feature Extractor 输出的 $d_2$ 维向量进行拼接 (Concatenation)。
    - 通过多层感知机 (MLP) 与 Dropout 层，最终输出 3 维的情感分类 logits (正向、负向、中性)。

## 💻 常用开发与构建命令

### 后端 (Python)
- **安装依赖**: `uv add <package>` 或同步 `uv sync`
- **运行类型检查**: `uvx pyright` (在提交代码前必须确保没有类型报错)
- **启动 API 服务**: `uv run uvicorn app.main:app --reload --port 8000`
- **运行训练脚本**: `uv run python scripts/train.py`

### 前端 (Vue.js + Vite)
- **安装依赖**: `npm install`
- **启动开发服务器**: `npm run dev` (通常运行在 5173 端口)
- **打包生产环境代码**: `npm run build`

## 📋 Claude 行为准则与代码规范

1.  **环境优先**: 执行任何 Python 脚本或安装依赖时，**必须**使用 `uv` 或 `uvx`，不要使用原生 `pip` 或 `conda`。
2.  **类型安全**: 后端 Python 代码必须包含严格的 Type Hints。在修改复杂逻辑后，主动使用 `uvx pyright` 检查错误。
3.  **工程化结构**: 
    - 算法与工程解耦：模型定义 (`models/`)、数据处理 (`dataset/`)、训练逻辑 (`train.py`) 和 Web 路由 (`api/`) 必须严格分层。
    - 前端组件化：在 Vue 开发中，利用 Vite 的特性，合理拆分 UI 组件，避免单个 `.vue` 文件过大。
4.  **科学的对比实验**: 在编写训练代码时，务必将准确率 (Accuracy)、精确率 (Precision)、召回率 (Recall) 和 F1 值作为核心评估指标，并支持生成混淆矩阵。
5.  **Git 工作流**: 在完成一个功能模块（如“完成数据清洗脚本”、“更新融合模型结构”）后，提醒用户进行 `git add` 和 `git commit`，并提供建议的 commit message。