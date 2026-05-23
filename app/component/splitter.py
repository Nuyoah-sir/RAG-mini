import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import get_settings


def _make_parent_splitter():
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        separators=["\n\n", "\n", "。", "？", "！", "；", " ", ""],
        length_function=len,
    )


def _make_child_splitter():
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=["\n\n", "\n", "。", "？", "！", "；", "：", " ", ""],
        length_function=len,
    )


def dual_zone_split(documents: list[Document]) -> tuple[list[Document], dict[str, Document]]:
    """双区分块：返回 (子块列表, parent_map)"""
    parent_splitter = _make_parent_splitter()
    child_splitter = _make_child_splitter()

    parent_chunks = parent_splitter.split_documents(documents)

    child_chunks: list[Document] = []
    parent_map: dict[str, Document] = {}

    for parent in parent_chunks:
        parent_id = uuid.uuid4().hex
        parent.metadata["parent_id"] = parent_id
        parent_map[parent_id] = parent

        # 在父块范围内切子块
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child_chunks.append(child)

    return child_chunks, parent_map


# 保留旧接口向后兼容
def create_splitter() -> RecursiveCharacterTextSplitter:
    """单区分块（旧接口，向后兼容）"""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.separators,
        length_function=len,
    )
