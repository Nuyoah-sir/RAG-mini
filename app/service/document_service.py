import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
)
from langchain_core.documents import Document

from app.core.logger import logger
from app.core.exceptions import DocumentLoadError, BuildError
from app.model.constants import SUPPORTED_EXTENSIONS
from app.component.splitter import dual_zone_split
from app.repository.vector_repo import build_vector_store


def _load_one(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    basename = os.path.basename(file_path)
    try:
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()
        elif ext in (".docx", ".doc"):
            return Docx2txtLoader(file_path).load()
        elif ext in (".txt", ".md"):
            return TextLoader(file_path, encoding="utf-8").load()
        else:
            logger.warning("跳过不支持的文件类型: %s", basename)
            return []
    except Exception as e:
        raise DocumentLoadError(basename, str(e))


def load_documents(file_paths: list[str]) -> list[Document]:
    docs = []
    for fp in file_paths:
        docs.extend(_load_one(fp))
    if not docs:
        raise DocumentLoadError("", "没有成功加载任何文档")
    logger.info("文档加载完成: %d 个文件 → %d 页", len(file_paths), len(docs))
    return docs


def build_kb(kb_name: str, file_paths: list[str]) -> dict:
    documents = load_documents(file_paths)
    child_chunks, parent_map = dual_zone_split(documents)
    logger.info("双区分块完成: %d 子块, %d 父块", len(child_chunks), len(parent_map))
    vs = build_vector_store(child_chunks, kb_name, parent_map=parent_map)
    return {"kb_name": kb_name, "documents": len(documents), "chunks": len(child_chunks)}
