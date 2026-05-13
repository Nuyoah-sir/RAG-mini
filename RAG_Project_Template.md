# 完全适配私有部署 Gemma-4-26B 的 RAG 项目模板（全远程 API 版）

## 最终技术栈（零本地大模型 / 重排序部署）

| 组件 | 选型 | 配置说明 |
|------|------|----------|
| 核心框架 | LangChain v0.2 | 统一接口，无缝对接私有大模型和第三方 API |
| 向量数据库 | FAISS | 本地轻量运行，无需额外服务 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 本地运行，中文效果最优轻量模型 |
| 重排序服务 | DeepSeek Rerank API | 远程调用，大幅提升检索精度 |
| 大语言模型 | 私有部署 Gemma-4-26B-A4B-IT | 通过 OpenAI 兼容接口调用 |
| Web 界面 | Gradio v4 | 一键生成交互式问答界面 |

## 项目结构

```
rag-project/
├── app.py              # 主程序入口
├── config.py           # 全局配置文件
├── requirements.txt    # 依赖列表
├── .env                # 敏感信息配置（API密钥）
├── data/               # 本地文档存放目录
└── vector_db/          # 本地向量数据库存储目录
```

## 第一步：环境配置

### 1.1 系统要求
- Python 3.10-3.12（推荐 3.11）
- 内存：≥4GB（纯 CPU 即可流畅运行）
- 网络：能访问 http://k8s.sgurad.com:7875 和 DeepSeek API

### 1.2 安装依赖

创建 `requirements.txt` 文件：

```
langchain>=0.3.0,<0.4.0
langchain-community>=0.3.0,<0.4.0
langchain-openai>=0.3.5,<0.4.0
langchain-deepseek>=0.1.2,<0.2.0
sentence-transformers==3.0.1
faiss-cpu==1.8.0
gradio==4.36.1
pypdf==4.2.0
python-docx==1.1.2
python-dotenv==1.0.1
```

安装命令：

```bash
# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装所有依赖
pip install -r requirements.txt
```

## 第二步：配置文件（核心修改）

### 2.1 创建 .env 文件（必须填写 DeepSeek API 密钥）

> ⚠️ 重要：将下面的 `你的DeepSeek API密钥` 替换为你实际的密钥，其他参数已预填完成

```env
# 私有部署Gemma大模型配置（已为你填好，无需修改）
OPENAI_API_BASE="http://k8s.sgurad.com:7875/v1"
OPENAI_API_KEY=""
LLM_MODEL_NAME="/home/sxqai/gemma-4-26b-a4b-it"

# DeepSeek重排序API配置（在这里填写你的DeepSeek API密钥）
DEEPSEEK_API_KEY="你的DeepSeek API密钥"
RERANKER_MODEL="deepseek-rerank"
```

### 2.2 创建 config.py 文件

```python
import os
from dotenv import load_dotenv

load_dotenv()

# 基础目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# 自动创建必要目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# 大模型API配置（从.env读取）
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# DeepSeek重排序API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

# 本地嵌入模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 文本分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "？", "！", "；", "：", " ", ""]

# 检索参数
TOP_K_RETRIEVE = 20  # 初步召回数量
TOP_K_RERANK = 3     # 重排序后保留数量

# LLM生成参数
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.95

# RAG提示模板（针对Gemma模型优化）
RAG_PROMPT_TEMPLATE = """
你是一个严谨的知识助手，必须**完全基于以下提供的上下文信息**回答用户问题。

【绝对规则】
1. 禁止编造任何上下文没有提到的信息
2. 如果上下文没有相关内容，直接回答："抱歉，我没有找到相关信息"
3. 回答要准确、简洁、分点清晰
4. 不要使用"根据上下文"、"根据提供的资料"等表述

【上下文信息】
{context}

【用户问题】
{question}

【你的回答】
"""
```

## 第三步：完整核心代码 app.py

```python
import os
import requests
import gradio as gr
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch

from config import *

# 全局变量
vector_store = None
qa_chain = None
reranker = None

# ========== 文档处理模块 ==========
def load_documents(file_paths):
    """加载PDF、Word、TXT、Markdown格式文档"""
    documents = []
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith((".docx", ".doc")):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith((".txt", ".md")):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                print(f"跳过不支持的文件: {os.path.basename(file_path)}")
                continue
            
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ 加载成功: {os.path.basename(file_path)} ({len(docs)}页)")
        except Exception as e:
            print(f"❌ 加载失败 {os.path.basename(file_path)}: {str(e)}")
    
    return documents

def split_documents(documents):
    """智能文本分块，保留语义完整性"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 文档已切分为 {len(chunks)} 个知识块")
    return chunks

# ========== 向量数据库模块 ==========
def build_vector_store(chunks):
    """构建本地FAISS向量数据库"""
    print("🔄 正在加载本地嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("🔄 正在构建向量索引...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_DIR)
    print(f"💾 向量数据库已保存到: {VECTOR_DB_DIR}")
    return vector_store

def load_vector_store():
    """加载已存在的本地向量数据库"""
    index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None
    
    print("🔄 正在加载现有向量数据库...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.load_local(
        VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store

# ========== 远程重排序模块（DeepSeek API） ==========
class DeepSeekReranker(BaseDocumentCompressor):
    """自定义DeepSeek重排序器，通过HTTP API调用"""

    model: str = "deepseek-rerank"
    api_key: str = ""
    top_n: int = 3

    def compress_documents(self, documents, query):
        """调用DeepSeek Rerank API对文档重排序"""
        if not documents or not self.api_key or self.api_key.startswith("你的"):
            return documents[:self.top_n]

        texts = [doc.page_content for doc in documents]
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": self.top_n
                },
                timeout=30
            )
            response.raise_for_status()
            results = response.json()["results"]
            ranked_indices = [r["index"] for r in sorted(
                results, key=lambda x: x["relevance_score"], reverse=True
            )]
            return [documents[i] for i in ranked_indices]
        except Exception as e:
            print(f"Rerank failed, fallback to original results: {str(e)}")
            return documents[:self.top_n]

def load_reranker():
    """加载DeepSeek远程重排序服务"""
    print("🔄 正在连接DeepSeek重排序API...")
    return DeepSeekReranker(
        model=RERANKER_MODEL,
        api_key=DEEPSEEK_API_KEY,
        top_n=TOP_K_RERANK
    )

# ========== 远程大模型模块（私有Gemma-4-26B） ==========
def load_llm():
    """连接私有部署的Gemma-4-26B大模型"""
    print(f"🔄 正在连接私有大模型: {LLM_MODEL_NAME}")
    
    llm = ChatOpenAI(
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL_NAME,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        streaming=True  # 支持流式输出
    )
    
    return llm

# ========== RAG问答链 ==========
def create_qa_chain(vector_store, llm):
    """创建带重排序的RAG问答链"""
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # 基础检索器：向量检索初步召回
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVE}
    )

    # 带DeepSeek重排序的检索器
    if reranker:
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
    else:
        retriever = base_retriever
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# ========== Gradio界面函数 ==========
def upload_files(files):
    """处理文件上传并构建向量库"""
    global vector_store, qa_chain
    
    if not files:
        return "请先上传文件"
    
    # files 是文件路径列表（Gradio 4.x 行为）
    file_paths = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as src:
            content = src.read()
        save_path = os.path.join(DATA_DIR, file_name)
        with open(save_path, "wb") as dst:
            dst.write(content)
        file_paths.append(save_path)
    
    # 加载和处理文档
    documents = load_documents(file_paths)
    if not documents:
        return "没有成功加载任何文档"
    
    chunks = split_documents(documents)
    
    # 构建向量库
    vector_store = build_vector_store(chunks)
    
    # 创建问答链
    qa_chain = create_qa_chain(vector_store, llm)
    
    return f"✅ 知识库构建完成！\n加载了 {len(documents)} 个文档\n切分为 {len(chunks)} 个知识块\n现在可以开始提问了"

def answer_question(question, history):
    """回答用户问题（支持流式输出）"""
    global qa_chain
    
    if not qa_chain:
        yield "请先上传文档并构建知识库"
        return
    
    if not question:
        yield "请输入您的问题"
        return
    
    try:
        answer = ""
        source_documents = []

        # 流式生成回答
        for chunk in qa_chain.stream({"query": question}):
            if "result" in chunk:
                answer += chunk["result"]
                yield answer
            if "source_documents" in chunk:
                source_documents = chunk["source_documents"]

        # 如果流式输出没有返回来源，再调用一次获取
        if not source_documents:
            result = qa_chain.invoke({"query": question})
            source_documents = result.get("source_documents", [])

        if source_documents:
            sources = []
            for i, doc in enumerate(source_documents):
                source = f"[{i+1}] "
                if "source" in doc.metadata:
                    source += f"{os.path.basename(doc.metadata['source'])}"
                if "page" in doc.metadata:
                    source += f" 第{doc.metadata['page']+1}页"
                sources.append(source)
            answer += "\n\n📚 参考来源：\n" + "\n".join(sources)
            yield answer
            
    except Exception as e:
        yield f"回答出错: {str(e)}"

def clear_history():
    """清空对话历史"""
    return None, None

# ========== 主程序 ==========
if __name__ == "__main__":
    # 预加载轻量模型和服务
    reranker = load_reranker()
    llm = load_llm()
    
    # 尝试加载已存在的向量库
    vector_store = load_vector_store()
    if vector_store:
        qa_chain = create_qa_chain(vector_store, llm)
        print("✅ 已加载现有向量数据库，可以直接提问")
    
    # 创建Gradio界面
    with gr.Blocks(title="私有Gemma RAG知识库助手") as demo:
        gr.Markdown("# 🤖 私有Gemma RAG知识库助手")
        gr.Markdown("上传PDF、Word、TXT或Markdown文件，构建您的私人知识库")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("🔨 构建知识库", variant="primary")
                status_text = gr.Textbox(label="状态", interactive=False)
                
                gr.Markdown("### 参数调整")
                temperature_slider = gr.Slider(
                    minimum=0, maximum=1, value=TEMPERATURE, step=0.1,
                    label="回答随机性（Temperature）"
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=TOP_K_RERANK, step=1,
                    label="参考文档数量"
                )
                
                clear_btn = gr.Button("🗑️ 清空对话")
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=600,
                    bubble_full_width=False
                )
                question_input = gr.Textbox(
                    label="输入您的问题",
                    placeholder="请输入您的问题，按回车发送"
                )
        
        # 绑定事件
        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[status_text]
        )
        
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot]
        ).then(lambda: "", None, [question_input])
        
        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, question_input]
        )
    
    # 启动服务
    print("🚀 服务已启动，请在浏览器中访问 http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
```

## 运行项目

1. 将上述 4 个文件（requirements.txt、.env、config.py、app.py）放在同一个文件夹中
2. 确保已在 .env 文件中填写了你的 DeepSeek API 密钥
3. 激活虚拟环境并运行主程序：

```bash
python app.py
```

- 程序会自动下载本地嵌入模型（首次运行需要等待）
- 浏览器会自动打开界面，地址为 http://localhost:7860

## 使用方法

- **上传文档**：点击 "上传文档" 按钮，选择您的 PDF、Word、TXT 或 Markdown 文件
- **构建知识库**：点击 "构建知识库" 按钮，等待处理完成
- **提问**：在输入框中输入问题，按回车发送（支持流式输出）
- **查看来源**：回答末尾会显示参考的文档和页码
- **调整参数**：可以调整回答随机性和参考文档数量

## 注意事项

- 私有大模型地址 http://k8s.sgurad.com:7875/v1 已预填，无需修改
- DeepSeek API 密钥必须填写，否则重排序功能无法使用
- 所有文档和向量数据库都存储在本地，不会上传到任何第三方服务器
- 首次运行会自动下载 BAAI/bge-small-zh-v1.5 嵌入模型（约 100MB）
