import pytest
from langchain_core.documents import Document
from app.component.splitter import dual_zone_split, create_splitter


class TestDualZoneSplit:
    """测试双区分块：父块+子块映射"""

    def test_returns_child_chunks_and_parent_map(self):
        docs = [Document(page_content="第一章\n\n这是一段测试内容。" * 20, metadata={"source": "test.txt"})]
        child_chunks, parent_map = dual_zone_split(docs)

        assert len(child_chunks) > 0
        assert len(parent_map) > 0
        # 每个子块都有 parent_id
        for chunk in child_chunks:
            assert "parent_id" in chunk.metadata
            pid = chunk.metadata["parent_id"]
            assert pid in parent_map

    def test_parent_map_values_are_documents(self):
        docs = [Document(page_content="第一章\n\n这是一段测试内容。" * 20, metadata={"source": "test.txt"})]
        child_chunks, parent_map = dual_zone_split(docs)

        for pid, parent_doc in parent_map.items():
            assert isinstance(parent_doc, Document)
            assert len(parent_doc.page_content) > 0

    def test_child_chunk_smaller_than_parent(self):
        docs = [Document(page_content="第一章\n\n这是一段测试内容。" * 20, metadata={"source": "test.txt"})]
        child_chunks, parent_map = dual_zone_split(docs)

        for child in child_chunks:
            pid = child.metadata["parent_id"]
            parent = parent_map[pid]
            # 子块不应超过父块大小
            assert len(child.page_content) <= len(parent.page_content)

    def test_single_short_document(self):
        """单段短文档：子块=父块"""
        docs = [Document(page_content="只有一句话。", metadata={"source": "short.txt"})]
        child_chunks, parent_map = dual_zone_split(docs)

        assert len(child_chunks) >= 1
        assert len(parent_map) >= 1
        child = child_chunks[0]
        parent = parent_map[child.metadata["parent_id"]]
        assert parent.page_content == child.page_content

    def test_preserves_original_metadata(self):
        docs = [Document(page_content="第一章\n\n这是一段测试内容。" * 20,
                         metadata={"source": "book.txt", "author": "test"})]
        child_chunks, parent_map = dual_zone_split(docs)

        for child in child_chunks:
            assert "source" in child.metadata
            # parent_id 是新增的，原始字段保留
            assert child.metadata.get("source") == "book.txt"

    def test_children_belong_to_one_parent_only(self):
        """每个子块只属于一个父块"""
        docs = [Document(page_content="第一章\n\n这是一段测试内容。" * 20, metadata={"source": "test.txt"})]
        child_chunks, parent_map = dual_zone_split(docs)

        # 所有子块的 parent_id 都存在于 parent_map 中
        parent_ids = set(c.metadata["parent_id"] for c in child_chunks)
        assert parent_ids.issubset(set(parent_map.keys()))

    def test_create_splitter_backward_compat(self):
        """旧接口 create_splitter 仍然可用"""
        splitter = create_splitter()
        docs = [Document(page_content="测试文档内容。" * 10, metadata={"source": "test.txt"})]
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
