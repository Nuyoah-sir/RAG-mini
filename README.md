# 多知识库 RAG 检索引擎

基于私有部署 Gemma-4-26B + FAISS 向量数据库 + DeepSeek 重排序的多知识库问答系统，支持 Gradio 交互界面、Python SDK 和 REST API 三种使用方式。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 大语言模型 | Gemma-4-26B-A4B-IT（私有部署） | OpenAI 兼容接口调用 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 本地运行，~100MB |
| 向量数据库 | FAISS | 本地轻量，零依赖服务 |
| 重排序 | DeepSeek Rerank API | 远程调用，提升检索精度 |
| 核心框架 | LangChain 0.3.x | 统一编排 |
| Web 界面 | Gradio 4.36.1 | 交互式问答 |
| REST API | FastAPI + uvicorn | 供外部系统调用 |
| GPU 加速 | PyTorch CUDA 2.5.1 | 可选，自动检测 |

## 项目结构

```
pythonProject/
├── app.py                    # Gradio 主程序（多知识库界面）
├── api.py                    # FastAPI REST 服务（端口 7880）
├── config.py                 # 全局配置 + 知识库管理
├── requirements.txt          # Python 依赖
├── .env                      # 敏感配置（API 密钥，不提交）
├── .env.example              # 配置模板
├── .gitignore
├── rag_sdk/                  # Python SDK 包
│   ├── __init__.py           # 导出 RAGEngine
│   ├── core.py               # 核心引擎实现
│   └── setup.py              # setuptools 打包配置
├── data/                     # 上传文档存放目录
└── vector_db/                # 向量数据库 + kb_config.json
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 内存 ≥ 4GB
- 网络可访问 DeepSeek API 和私有 LLM 地址

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

GPU 用户额外安装：

```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. 配置

复制 `.env.example` 为 `.env`，填写你的配置：

```env
OPENAI_API_BASE=""
OPENAI_API_KEY=""
LLM_MODEL_NAME=""
DEEPSEEK_API_KEY="你的DeepSeek API密钥"
RERANKER_MODEL="deepseek-rerank"
```

### 4. 启动

**Gradio 交互界面**（端口 7860）：

```bash
python app.py
```

**REST API 服务**（端口 7880）：

```bash
python api.py
```

## 三种使用方式

### 方式一：Gradio Web 界面

浏览器访问 `http://localhost:7860`，可以：
- 创建/删除多个知识库
- 上传 PDF、Word、TXT、Markdown 文档构建索引
- 选择知识库进行问答，流式输出答案及参考来源

### 方式二：Python SDK

```python
from rag_sdk import RAGEngine

engine = RAGEngine(
    llm_base_url="",
    llm_model="",
    deepseek_api_key="sk-xxx",
)

# 创建知识库并构建索引
engine.create_kb("论文资料")
engine.build_kb("论文资料", ["/path/to/doc1.pdf", "/path/to/doc2.docx"])

# 列出知识库
for kb in engine.list_kbs():
    print(kb.name, kb.has_index)

# 问答
result = engine.ask("论文资料", "系统用了哪些技术栈？")
print(result.answer)
for src in result.sources:
    print(f"  [{src.file}] {src.snippet[:100]}...")
```

### 方式三：REST API（供 Java / 其他语言调用）

```java
// Java 调用示例
HttpClient client = HttpClient.newHttpClient();
String json = """
    {"kb_name": "论文资料", "question": "系统用了哪些技术栈？"}
    """;

HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("http://your-server:7880/api/ask"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(json))
    .build();

HttpResponse<String> response = client.send(request,
    HttpResponse.BodyHandlers.ofString());
System.out.println(response.body());
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/kb/list` | 列出所有知识库 |
| POST | `/api/kb/create` | 创建知识库 `{"kb_name": "xxx"}` |
| POST | `/api/kb/{name}/build` | 构建索引 `{"kb_name": "x", "file_paths": [...]}` |
| POST | `/api/ask` | 问答 `{"kb_name": "x", "question": "..."}` |
| DELETE | `/api/kb/{name}` | 删除知识库 |
| GET | `/api/health` | 健康检查 |

## 多知识库设计

- 每个知识库独立存储为一个 FAISS 索引目录，互不干扰
- 知识库名称到目录使用 MD5 映射，避免中文路径兼容问题
- 映射关系保存在 `vector_db/kb_config.json`
- 索引懒加载：首次提问时才加载对应知识库到内存

## GPU 加速

程序自动检测 CUDA 可用性（`device="auto"`），embedding 模型会自动使用 GPU。RTX 4060 等消费级显卡可显著加速文档索引构建和检索。

## SDK 安装

```bash
pip install -e rag_sdk/
```

## 依赖版本说明

部分依赖有版本约束以防止兼容性问题：

- `numpy<2`：FAISS 1.8.0 编译依赖 NumPy 1.x ABI
- `starlette<1.0` + `jinja2==3.1.4`：Gradio 4.36.1 兼容
- `langchain-huggingface<1.0`：保持与 LangChain 0.3.x 兼容
- `torch` 需从 pytorch.org 安装 CUDA 版本，PyPI 仅有 CPU 版

## License

MIT
