import os
import json
import requests
import gradio as gr
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch

from config import *

# 全局状态：多知识库
kb_stores = {}   # kb_name -> FAISS vectorstore
kb_chains = {}   # kb_name -> RetrievalQA chain
embedding_model = None
reranker = None
llm = None

def get_kb_choices():
    """获取知识库下拉选项"""
    kbs = list_knowledge_bases()
    choices = []
    for kb in kbs:
        label = kb["name"]
        if kb["has_index"]:
            label += " [已就绪]"
        else:
            label += " [空]"
        choices.append((label, kb["name"]))
    return choices


# ========== 文档处理模块（不变） ==========
def load_documents(file_paths):
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
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("🔄 正在加载本地嵌入模型...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return embedding_model


def build_vector_store(chunks, kb_name):
    embeddings = get_embedding_model()
    kb_path = get_kb_path(kb_name)
    print(f"🔄 正在为 [{kb_name}] 构建向量索引...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(kb_path)
    print(f"💾 [{kb_name}] 向量数据库已保存到: {kb_path}")
    return vs


def load_vector_store(kb_name):
    index_path = os.path.join(get_kb_path(kb_name), "index.faiss")
    if not os.path.exists(index_path):
        return None
    print(f"🔄 正在加载 [{kb_name}] 向量数据库...")
    embeddings = get_embedding_model()
    return FAISS.load_local(
        get_kb_path(kb_name), embeddings, allow_dangerous_deserialization=True
    )


# ========== 远程重排序模块 ==========
class DeepSeekReranker(BaseDocumentCompressor):
    model: str = "deepseek-rerank"
    api_key: str = ""
    top_n: int = 3

    def compress_documents(self, documents, query, callbacks=None, **kwargs):
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


def get_reranker():
    global reranker
    if reranker is None:
        print("🔄 正在连接DeepSeek重排序API...")
        reranker = DeepSeekReranker(
            model=RERANKER_MODEL,
            api_key=DEEPSEEK_API_KEY,
            top_n=TOP_K_RERANK
        )
    return reranker


# ========== 远程大模型模块 ==========
def get_llm():
    global llm
    if llm is None:
        print(f"🔄 正在连接私有大模型: {LLM_MODEL_NAME}")
        llm = ChatOpenAI(
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY or "not-needed",
            model=LLM_MODEL_NAME,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True
        )
    return llm


# ========== RAG问答链 ==========
def create_qa_chain(vs, kb_name):
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    base_retriever = vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
    r = get_reranker()
    if r:
        retriever = ContextualCompressionRetriever(
            base_compressor=r,
            base_retriever=base_retriever
        )
    else:
        retriever = base_retriever
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


def get_or_load_kb(kb_name):
    """获取或懒加载知识库的问答链"""
    if not kb_name:
        return None
    if kb_name not in kb_chains:
        vs = load_vector_store(kb_name)
        if vs:
            kb_stores[kb_name] = vs
            kb_chains[kb_name] = create_qa_chain(vs, kb_name)
        else:
            return None
    return kb_chains[kb_name]


# ========== Gradio界面函数 ==========
def refresh_kb_list():
    """刷新知识库下拉列表"""
    choices = get_kb_choices()
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def create_kb(kb_name):
    """创建新知识库"""
    if not kb_name or not kb_name.strip():
        return "请输入知识库名称", refresh_kb_list()
    kb_name = kb_name.strip()
    # 验证名称合法性
    if any(c in kb_name for c in r'\/:*?"<>|'):
        return "知识库名称不能包含特殊字符", refresh_kb_list()
    kb_path = get_kb_path(kb_name)
    if os.path.exists(os.path.join(kb_path, "index.faiss")):
        return f"知识库 [{kb_name}] 已存在", refresh_kb_list()
    os.makedirs(kb_path, exist_ok=True)
    return f"✅ 知识库 [{kb_name}] 创建成功，请上传文档", refresh_kb_list()


def delete_kb(kb_name):
    """删除知识库"""
    if not kb_name:
        return "请先选择知识库", refresh_kb_list()
    import shutil
    kb_path = get_kb_path(kb_name)
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)
    kb_stores.pop(kb_name, None)
    kb_chains.pop(kb_name, None)
    return f"🗑️ 知识库 [{kb_name}] 已删除", refresh_kb_list()


def upload_files(files, kb_name):
    """处理文件上传并构建向量库"""
    if not kb_name:
        return "请先选择或创建知识库"

    if not files:
        return "请先上传文件"

    # 保存上传的文件
    file_paths = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as src:
            content = src.read()
        save_path = os.path.join(DATA_DIR, file_name)
        with open(save_path, "wb") as dst:
            dst.write(content)
        file_paths.append(save_path)

    documents = load_documents(file_paths)
    if not documents:
        return "没有成功加载任何文档"

    chunks = split_documents(documents)
    vs = build_vector_store(chunks, kb_name)

    # 更新全局状态
    kb_stores[kb_name] = vs
    kb_chains[kb_name] = create_qa_chain(vs, kb_name)

    return f"✅ [{kb_name}] 知识库构建完成！\n加载 {len(documents)} 个文档，切分为 {len(chunks)} 个知识块\n现在可以开始提问了"


def answer_question(question, history, kb_name):
    """回答用户问题"""
    if not kb_name:
        history = history or []
        history.append([question, "请先选择知识库"])
        yield history
        return

    chain = get_or_load_kb(kb_name)
    if not chain:
        history = history or []
        history.append([question, f"知识库 [{kb_name}] 为空，请先上传文档"])
        yield history
        return

    if not question:
        return

    history = history or []
    history.append([question, ""])

    try:
        answer = ""
        source_documents = []

        for chunk in chain.stream({"query": question}):
            if "result" in chunk:
                answer += chunk["result"]
                history[-1][1] = answer
                yield history
            if "source_documents" in chunk:
                source_documents = chunk["source_documents"]

        if not source_documents:
            result = chain.invoke({"query": question})
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
            history[-1][1] = answer
            yield history

    except Exception as e:
        history[-1][1] = f"回答出错: {str(e)}"
        yield history


def clear_history():
    return None, None


# ========== 主程序 ==========
if __name__ == "__main__":
    # Monkey-patch: 修复 gradio_client 无法处理 JSON Schema boolean 值的 bug
    import gradio_client.utils as gcu
    _orig_get_type = gcu.get_type
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_get_type(schema)
    gcu.get_type = _patched_get_type
    _orig_json_schema = gcu._json_schema_to_python_type
    def _patched_json_schema(schema, defs=None):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_json_schema(schema, defs)
    gcu._json_schema_to_python_type = _patched_json_schema

    # 预热模型
    get_embedding_model()
    get_reranker()
    get_llm()

    # 预加载已有知识库
    existing_kbs = list_knowledge_bases()
    for kb in existing_kbs:
        if kb["has_index"]:
            print(f"✅ 发现已有知识库: [{kb['name']}]")
            get_or_load_kb(kb["name"])

    # 创建Gradio界面
    with gr.Blocks(title="多知识库 RAG 助手") as demo:
        gr.Markdown("# 🤖 多知识库 RAG 助手")
        gr.Markdown("创建独立知识库，分别管理不同领域的文档")

        # ---- 知识库管理区 ----
        with gr.Row():
            kb_dropdown = gr.Dropdown(
                label="当前知识库",
                choices=get_kb_choices(),
                value=existing_kbs[0]["name"] if existing_kbs else None,
                scale=3,
                interactive=True
            )
            new_kb_name = gr.Textbox(
                label="新建知识库名称",
                placeholder="输入名称后点创建",
                scale=2
            )
            create_kb_btn = gr.Button("📁 创建", scale=1)
            delete_kb_btn = gr.Button("🗑️ 删除", scale=1, variant="stop")
        kb_status = gr.Textbox(label="操作状态", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="上传文档到当前知识库",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("🔨 构建知识库", variant="primary")
                upload_status = gr.Textbox(label="构建状态", interactive=False)

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

        # ---- 绑定事件 ----
        # KB 管理
        create_kb_btn.click(
            fn=create_kb,
            inputs=[new_kb_name],
            outputs=[kb_status, kb_dropdown]
        )
        delete_kb_btn.click(
            fn=delete_kb,
            inputs=[kb_dropdown],
            outputs=[kb_status, kb_dropdown]
        )

        # 文档上传
        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload, kb_dropdown],
            outputs=[upload_status]
        )

        # 问答
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, chatbot, kb_dropdown],
            outputs=[chatbot]
        ).then(lambda: "", None, [question_input])

        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, question_input]
        )

    print("🚀 服务已启动，请在浏览器中访问 http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
