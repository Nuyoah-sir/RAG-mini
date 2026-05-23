import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.core.config import get_settings, get_kb_path
from app.core.logger import logger
from app.component.embedding import get_embedding

# 全局向量库缓存
_stores: dict[str, FAISS] = {}
_parent_maps: dict[str, dict[str, Document]] = {}


def build_vector_store(chunks: list[Document], kb_name: str,
                       parent_map: dict[str, Document] | None = None) -> FAISS:
    settings = get_settings()
    embeddings = get_embedding()
    kb_path = get_kb_path(settings, kb_name)
    logger.info("构建向量索引: [%s], %d chunks", kb_name, len(chunks))
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(kb_path))
    logger.info("向量索引已保存: %s", kb_path)
    _stores[kb_name] = vs

    # 保存父块映射
    if parent_map:
        pm_path = kb_path / "parent_map.pkl"
        with open(pm_path, "wb") as f:
            pickle.dump(parent_map, f)
        logger.info("父块映射已保存: %s (%d entries)", pm_path, len(parent_map))

    # 同时构建 BM25 关键词索引
    if settings.bm25_enabled:
        from app.component.bm25 import BM25Manager
        bm = BM25Manager()
        bm.build(chunks)
        bm.save(kb_path / "bm25_index.pkl")

    return vs


def load_vector_store(kb_name: str) -> FAISS | None:
    if kb_name in _stores:
        return _stores[kb_name]

    settings = get_settings()
    kb_path = get_kb_path(settings, kb_name)
    index_file = kb_path / "index.faiss"
    if not index_file.exists():
        return None

    logger.info("加载向量索引: [%s]", kb_name)
    embeddings = get_embedding()
    vs = FAISS.load_local(
        str(kb_path), embeddings, allow_dangerous_deserialization=True
    )
    _stores[kb_name] = vs

    # 加载父块映射
    pm_path = kb_path / "parent_map.pkl"
    if pm_path.exists():
        with open(pm_path, "rb") as f:
            _parent_maps[kb_name] = pickle.load(f)
        logger.info("父块映射已加载: [%s] (%d entries)", kb_name, len(_parent_maps[kb_name]))

    # 同时加载 BM25 索引
    if settings.bm25_enabled:
        from app.component.bm25 import get_bm25
        get_bm25(kb_name, kb_path)

    return vs


def get_parent_map(kb_name: str) -> dict[str, Document] | None:
    """获取知识库的父块映射"""
    return _parent_maps.get(kb_name)


def evict_store(kb_name: str) -> None:
    _stores.pop(kb_name, None)
    _parent_maps.pop(kb_name, None)
    # 同时清除 BM25
    from app.component.bm25 import evict_bm25
    evict_bm25(kb_name)


def get_store_stats(kb_name: str) -> dict | None:
    vs = _stores.get(kb_name)
    if vs is None:
        return None
    n_chunks = vs.index.ntotal
    sources = set()
    for k in vs.docstore._dict:
        doc = vs.docstore._dict[k]
        if hasattr(doc, "metadata") and "source" in doc.metadata:
            sources.add(doc.metadata["source"])
    return {"n_docs": len(sources), "n_chunks": n_chunks}
