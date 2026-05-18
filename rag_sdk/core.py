"""
RAG SDK — 多知识库检索引擎

使用示例:
    from rag_sdk import RAGEngine

    engine = RAGEngine(
        llm_base_url="",
        llm_model="",
        deepseek_api_key="sk-xxx",
    )

    # 创建知识库并构建索引
    engine.create_kb("论文资料")
    engine.build_kb("论文资料", ["/path/to/doc1.pdf", "/path/to/doc2.docx"])

    # 问答
    result = engine.ask("论文资料", "系统用了哪些技术栈？")
    print(result.answer)
    print(result.sources)
"""
import os
import shutil
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import requests
import torch
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


# ---- 数据结构 ----
@dataclass
class Source:
    file: str = ""
    page: Optional[int] = None
    snippet: str = ""

@dataclass
class AskResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    kb_name: str = ""

@dataclass
class KBInfo:
    name: str
    has_index: bool = False


# ---- 默认配置 ----
DEFAULT_PROMPT = """
你是一个严谨的知识助手，必须**完全基于以下提供的上下文信息**回答用户问题。

【绝对规则】
1. 禁止编造任何上下文没有提到的信息
2. 如果上下文没有相关内容，直接回答："抱歉，我没有找到相关信息"
3. 回答要准确、简洁、分点清晰

【上下文信息】
{context}

【用户问题】
{question}

【你的回答】
"""

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "？", "！", "；", "：", " ", ""]
DEFAULT_TOP_K_RETRIEVE = 20
DEFAULT_TOP_K_RERANK = 3


# ---- 自定义重排序器 ----
class _DeepSeekReranker(BaseDocumentCompressor):
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
                json={"model": self.model, "query": query,
                       "documents": texts, "top_n": self.top_n},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()["results"]
            ranked = [r["index"] for r in sorted(
                results, key=lambda x: x["relevance_score"], reverse=True)]
            return [documents[i] for i in ranked]
        except Exception as e:
            print(f"Rerank failed: {e}")
            return documents[:self.top_n]


# ---- 核心引擎 ----
class RAGEngine:
    """多知识库 RAG 引擎"""

    def __init__(
        self,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str = "",
        deepseek_api_key: str = "",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        vector_db_dir: str = "./vector_db",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE,
        top_k_rerank: int = DEFAULT_TOP_K_RERANK,
        prompt_template: str = DEFAULT_PROMPT,
        device: str = "auto",
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.deepseek_api_key = deepseek_api_key
        self.embedding_model_name = embedding_model
        self.vector_db_dir = vector_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.prompt_template = prompt_template

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        os.makedirs(vector_db_dir, exist_ok=True)
        self.kb_config_file = os.path.join(vector_db_dir, "kb_config.json")

        # 懒加载的组件
        self._embedding = None
        self._llm = None
        self._reranker = None
        self._stores: dict[str, FAISS] = {}
        self._chains: dict[str, RetrievalQA] = {}

    # ---- 内部方法 ----
    def _safe_dirname(self, kb_name: str) -> str:
        return "kb_" + hashlib.md5(kb_name.encode("utf-8")).hexdigest()[:12]

    def _load_kb_config(self) -> dict:
        if os.path.exists(self.kb_config_file):
            with open(self.kb_config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_kb_config(self, config: dict):
        with open(self.kb_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _get_kb_path(self, kb_name: str) -> str:
        config = self._load_kb_config()
        safe = self._safe_dirname(kb_name)
        if safe not in config:
            config[safe] = kb_name
            self._save_kb_config(config)
        path = os.path.join(self.vector_db_dir, safe)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_embedding(self):
        if self._embedding is None:
            self._embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embedding

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                base_url=self.llm_base_url,
                api_key=self.llm_api_key or "not-needed",
                model=self.llm_model,
                temperature=0.1,
                streaming=False
            )
        return self._llm

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = _DeepSeekReranker(
                api_key=self.deepseek_api_key,
                top_n=self.top_k_rerank
            )
        return self._reranker

    def _load_store(self, kb_name: str) -> Optional[FAISS]:
        index_path = os.path.join(self._get_kb_path(kb_name), "index.faiss")
        if not os.path.exists(index_path):
            return None
        return FAISS.load_local(
            self._get_kb_path(kb_name),
            self._get_embedding(),
            allow_dangerous_deserialization=True
        )

    def _create_chain(self, vs: FAISS):
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        base_retriever = vs.as_retriever(
            search_kwargs={"k": self.top_k_retrieve}
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=self._get_reranker(),
            base_retriever=base_retriever
        )
        return RetrievalQA.from_chain_type(
            llm=self._get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def _get_or_load(self, kb_name: str):
        if kb_name not in self._chains:
            vs = self._load_store(kb_name)
            if vs:
                self._stores[kb_name] = vs
                self._chains[kb_name] = self._create_chain(vs)
            else:
                return None
        return self._chains[kb_name]

    # ---- 公开 API ----
    def list_kbs(self) -> list[KBInfo]:
        """列出所有知识库"""
        config = self._load_kb_config()
        kbs = []
        for safe, name in config.items():
            kb_path = os.path.join(self.vector_db_dir, safe)
            if os.path.isdir(kb_path):
                has_idx = os.path.exists(os.path.join(kb_path, "index.faiss"))
                kbs.append(KBInfo(name=name, has_index=has_idx))
        return kbs

    def create_kb(self, kb_name: str):
        """创建新知识库"""
        if not kb_name or not kb_name.strip():
            raise ValueError("知识库名称不能为空")
        self._get_kb_path(kb_name.strip())

    def delete_kb(self, kb_name: str):
        """删除知识库"""
        kb_path = self._get_kb_path(kb_name)
        if os.path.exists(kb_path):
            shutil.rmtree(kb_path)
        self._stores.pop(kb_name, None)
        self._chains.pop(kb_name, None)

    def build_kb(self, kb_name: str, file_paths: list[str]) -> dict:
        """构建知识库索引"""
        documents = []
        for fp in file_paths:
            try:
                if fp.endswith(".pdf"):
                    loader = PyPDFLoader(fp)
                elif fp.endswith((".docx", ".doc")):
                    loader = Docx2txtLoader(fp)
                elif fp.endswith((".txt", ".md")):
                    loader = TextLoader(fp, encoding="utf-8")
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"加载失败 {fp}: {e}")

        if not documents:
            raise ValueError("没有成功加载任何文档")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=DEFAULT_SEPARATORS,
            length_function=len
        )
        chunks = splitter.split_documents(documents)

        vs = FAISS.from_documents(chunks, self._get_embedding())
        vs.save_local(self._get_kb_path(kb_name))
        self._stores[kb_name] = vs
        self._chains[kb_name] = self._create_chain(vs)

        return {"documents": len(documents), "chunks": len(chunks)}

    def ask(self, kb_name: str, question: str) -> AskResult:
        """向指定知识库提问"""
        chain = self._get_or_load(kb_name)
        if not chain:
            raise ValueError(f"知识库 [{kb_name}] 不存在或为空")

        result = chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        sources = []
        for doc in source_docs:
            src = Source(snippet=doc.page_content[:200])
            if "source" in doc.metadata:
                src.file = os.path.basename(doc.metadata["source"])
            if "page" in doc.metadata:
                src.page = doc.metadata["page"] + 1
            sources.append(src)

        return AskResult(answer=answer, sources=sources, kb_name=kb_name)
