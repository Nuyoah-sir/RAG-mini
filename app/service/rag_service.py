import os
import hashlib
from typing import Iterator, Optional

from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from app.core.config import get_settings, get_kb_path
from app.core.logger import logger
from app.core.exceptions import KBNotFoundError, AskError
from app.model.dto import AskResponse, Source
from app.component.llm import get_llm
from app.component.reranker import get_reranker
from app.repository.vector_repo import load_vector_store, get_parent_map as get_parent_map_from_repo

_chains: dict[str, RetrievalQA] = {}


class HybridRetriever(BaseRetriever):
    """融合 FAISS（语义）+ BM25（关键词）的检索器，支持双区父块解析"""
    faiss_retriever: BaseRetriever
    bm25_manager: object  # BM25Manager, avoid import cycle
    top_k: int = 20
    parent_map: dict | None = None  # 新增：父块映射

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        # FAISS 语义检索
        dense_docs = self.faiss_retriever._get_relevant_documents(query, run_manager=run_manager)

        # BM25 关键词检索
        sparse_docs = []
        bm = self.bm25_manager
        if bm is not None and hasattr(bm, 'search'):
            sparse_docs = bm.search(query, self.top_k)

        # 合并去重（用 page_content 前 200 字符的 hash 做为 key）
        seen = {}
        for doc in dense_docs:
            key = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
            seen[key] = doc
        for doc in sparse_docs:
            key = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
            if key not in seen:
                seen[key] = doc

        child_docs = list(seen.values())
        logger.debug("混合检索: FAISS=%d BM25=%d 合并=%d", len(dense_docs), len(sparse_docs), len(child_docs))

        # 双区父块解析：子块 → 父块 + parent_id 去重
        if self.parent_map:
            resolved = {}
            for child in child_docs:
                pid = child.metadata.get("parent_id")
                if pid and pid in self.parent_map:
                    if pid not in resolved:
                        resolved[pid] = self.parent_map[pid]
                else:
                    # 无 parent_id 的旧索引，直接保留
                    resolved[child.metadata.get("parent_id") or id(child)] = child
            return list(resolved.values())

        return child_docs


def _create_chain(kb_name: str) -> RetrievalQA:
    settings = get_settings()
    vs = load_vector_store(kb_name)
    if vs is None:
        raise KBNotFoundError(kb_name)

    prompt = PromptTemplate(
        template=settings.rag_prompt_template,
        input_variables=["context", "question"],
    )

    # FAISS base retriever
    faiss_retriever = vs.as_retriever(search_kwargs={"k": settings.top_k_retrieve})

    # BM25 manager (may be None if disabled)
    bm = None
    if settings.bm25_enabled:
        from app.component.bm25 import get_bm25
        kb_path = get_kb_path(settings, kb_name)
        bm = get_bm25(kb_name, kb_path)

    # 混合检索器：融合 FAISS + BM25，再经过重排序
    base_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_manager=bm,
        top_k=settings.top_k_retrieve,
        parent_map=get_parent_map_from_repo(kb_name),  # 新增
    )
    reranker = get_reranker()
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def _get_or_load_chain(kb_name: str) -> RetrievalQA:
    if kb_name not in _chains:
        _chains[kb_name] = _create_chain(kb_name)
    return _chains[kb_name]


def evict_chain(kb_name: str) -> None:
    _chains.pop(kb_name, None)


def ask(kb_name: str, question: str) -> AskResponse:
    try:
        chain = _get_or_load_chain(kb_name)
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

        return AskResponse(answer=answer, sources=sources, kb_name=kb_name)
    except KBNotFoundError:
        raise
    except Exception as e:
        logger.exception("问答失败 [%s]", kb_name)
        raise AskError(kb_name, str(e))


def ask_stream(kb_name: str, question: str) -> Iterator[dict]:
    try:
        chain = _get_or_load_chain(kb_name)
        answer = ""
        for chunk in chain.stream({"query": question}):
            if "result" in chunk:
                answer += chunk["result"]
                yield {"type": "token", "data": chunk["result"]}
        yield {"type": "done", "answer": answer}
    except KBNotFoundError:
        raise
    except Exception as e:
        logger.exception("流式问答失败 [%s]", kb_name)
        raise AskError(kb_name, str(e))


def preload_all() -> None:
    from app.repository.kb_meta_repo import list_kbs

    for kb in list_kbs():
        if kb["has_index"]:
            try:
                _get_or_load_chain(kb["name"])
                logger.info("预加载知识库: [%s]", kb["name"])
            except Exception as e:
                logger.warning("预加载知识库失败 [%s]: %s", kb["name"], e)
