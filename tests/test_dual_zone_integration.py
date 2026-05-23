# tests/test_dual_zone_integration.py
import os
from langchain_core.documents import Document
from app.component.splitter import dual_zone_split
from app.repository.vector_repo import build_vector_store, load_vector_store, get_parent_map, evict_store


class TestDualZoneIntegration:
    """端到端：分块 → 建索引 → 加载 → 父块映射"""

    def test_full_cycle(self):
        kb_name = "_test_dz_" + os.urandom(4).hex()
        docs = [Document(
            page_content="第一章 引言\n\n" + "这是测试内容。" * 50,
            metadata={"source": "test.txt"}
        )]

        # 1. 双区分块
        child_chunks, parent_map = dual_zone_split(docs)
        assert len(child_chunks) > 0
        assert len(parent_map) > 0

        # 2. 建索引
        vs = build_vector_store(child_chunks, kb_name, parent_map=parent_map)
        assert vs is not None

        # 3. 加载并验证 parent_map
        evict_store(kb_name)  # 清除缓存强制从磁盘加载
        vs2 = load_vector_store(kb_name)
        pm = get_parent_map(kb_name)
        assert pm is not None
        assert len(pm) == len(parent_map)
        for pid, parent_doc in pm.items():
            assert pid in parent_map

        # 4. 清理
        evict_store(kb_name)
        from app.core.config import get_settings, get_kb_path
        settings = get_settings()
        kb_path = get_kb_path(settings, kb_name)
        import shutil
        if kb_path.exists():
            shutil.rmtree(kb_path)

    def test_no_parent_map_still_works(self):
        """无 parent_map 的旧索引也能正常加载"""
        kb_name = "_test_dz_old_" + os.urandom(4).hex()
        docs = [Document(page_content="旧格式文档内容。" * 30, metadata={"source": "old.txt"})]
        from app.component.splitter import create_splitter
        splitter = create_splitter()
        chunks = splitter.split_documents(docs)

        vs = build_vector_store(chunks, kb_name)  # 不传 parent_map
        assert vs is not None

        evict_store(kb_name)
        vs2 = load_vector_store(kb_name)
        pm = get_parent_map(kb_name)
        assert pm is None  # 旧索引无 parent_map

        evict_store(kb_name)
        from app.core.config import get_settings, get_kb_path
        settings = get_settings()
        kb_path = get_kb_path(settings, kb_name)
        import shutil
        if kb_path.exists():
            shutil.rmtree(kb_path)
