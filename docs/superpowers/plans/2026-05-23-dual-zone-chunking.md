# Dual-Zone Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement parent-child dual-zone chunking — small child chunks indexed for precise retrieval, large parent chunks returned for coherent LLM context.

**Architecture:** DualZoneSplitter produces (child_chunks, parent_map). Child chunks go to FAISS/BM25 for search. parent_map persists alongside the index. At retrieval, HybridRetriever resolves child hits to parent docs via parent_map, deduplicates by parent_id, and returns parent chunks to the reranker/LLM.

**Tech Stack:** LangChain 0.3.x, FAISS, rank_bm25, Python 3.11+

---

### Task 1: Add configuration fields

**Files:**
- Modify: `app/core/config.py:32-35`

- [ ] **Step 1: Add parent/child chunk config to Settings**

```python
# 将第 32-35 行替换为:
    # ---- 分块 ----
    chunk_size: int = 500          # 保留向后兼容
    chunk_overlap: int = 50
    separators: list = field(default_factory=lambda: ["\n\n", "\n", "。", "？", "！", "；", "：", " ", ""])

    # ---- 双区分块 ----
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 100
```

- [ ] **Step 2: Verify config loads correctly**

```bash
python -c "from app.core.config import get_settings; s=get_settings(); print(s.parent_chunk_size, s.child_chunk_size)"
```
Expected: `1200 400`

- [ ] **Step 3: Commit**

```bash
git add app/core/config.py
git commit -m "feat: add parent/child chunk size config for dual-zone chunking"
```

---

### Task 2: Rewrite splitter with DualZoneSplitter

**Files:**
- Create: `tests/test_splitter.py`
- Modify: `app/component/splitter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_splitter.py
import pytest
from langchain_core.documents import Document
from app.component.splitter import dual_zone_split


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_splitter.py -v
```
Expected: FAIL — `dual_zone_split` not defined

- [ ] **Step 3: Implement DualZoneSplitter**

```python
# app/component/splitter.py — 完整替换
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_splitter.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/component/splitter.py tests/test_splitter.py
git commit -m "feat: add DualZoneSplitter with parent-child chunking"
```

---

### Task 3: Update vector_repo to persist parent_map

**Files:**
- Modify: `app/repository/vector_repo.py:14-31`
- Modify: `app/repository/vector_repo.py:34-56`

- [ ] **Step 1: Update build_vector_store signature and save parent_map**

```python
# vector_repo.py — 修改 build_vector_store
import pickle

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
```

- [ ] **Step 2: Update load_vector_store to load parent_map**

```python
# vector_repo.py — 新增 _parent_maps 缓存 + 修改 load_vector_store
import pickle

_parent_maps: dict[str, dict[str, Document]] = {}


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
```

注：`import pickle` 只在文件顶部加一次（所有修改共用一个 import）。

- [ ] **Step 3: Verify vector store still works with old (no parent_map) flow**

```bash
python -c "from app.repository.vector_repo import load_vector_store; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/repository/vector_repo.py
git commit -m "feat: persist and load parent_map alongside FAISS index"
```

---

### Task 4: Wire parent_map through document_service

**Files:**
- Modify: `app/service/document_service.py:43-49`

- [ ] **Step 1: Update build_kb to use dual_zone_split**

```python
# document_service.py — 修改 build_kb
from app.component.splitter import dual_zone_split

def build_kb(kb_name: str, file_paths: list[str]) -> dict:
    documents = load_documents(file_paths)
    child_chunks, parent_map = dual_zone_split(documents)
    logger.info("双区分块完成: %d 子块, %d 父块", len(child_chunks), len(parent_map))
    vs = build_vector_store(child_chunks, kb_name, parent_map=parent_map)
    return {"kb_name": kb_name, "documents": len(documents), "chunks": len(child_chunks)}
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from app.service.document_service import build_kb; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/service/document_service.py
git commit -m "feat: wire dual_zone_split into build_kb pipeline"
```

---

### Task 5: Update HybridRetriever for child-to-parent resolution

**Files:**
- Modify: `app/service/rag_service.py:22-53` (HybridRetriever)
- Modify: `app/service/rag_service.py:56-93` (_create_chain)

- [ ] **Step 1: Update HybridRetriever to resolve child→parent**

```python
# rag_service.py — 修改 HybridRetriever 类
class HybridRetriever(BaseRetriever):
    """融合 FAISS（语义）+ BM25（关键词）的检索器，支持双区父块解析"""
    faiss_retriever: BaseRetriever
    bm25_manager: object
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

        # 合并去重
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
```

- [ ] **Step 2: Update _create_chain to pass parent_map to retriever**

```python
# rag_service.py — 修改 _create_chain 中的 HybridRetriever 实例化
    # 混合检索器：融合 FAISS + BM25，再经过重排序
    base_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_manager=bm,
        top_k=settings.top_k_retrieve,
        parent_map=get_parent_map_from_repo(kb_name),  # 新增
    )
```

并在文件顶部添加导入：
```python
from app.repository.vector_repo import load_vector_store, get_parent_map as get_parent_map_from_repo
```

（替换原有的 `from app.repository.vector_repo import load_vector_store`）

- [ ] **Step 3: Verify import chain works**

```bash
python -c "from app.service.rag_service import ask; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/service/rag_service.py
git commit -m "feat: resolve child chunks to parent chunks in HybridRetriever"
```

---

### Task 6: Integration smoke test

**Files:**
- Create: `tests/test_dual_zone_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_dual_zone_integration.py
import tempfile
import os
from pathlib import Path
from app.component.splitter import dual_zone_split
from app.repository.vector_repo import build_vector_store, load_vector_store, get_parent_map, evict_store
from langchain_core.documents import Document


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
```

- [ ] **Step 2: Run integration tests**

```bash
python -m pytest tests/test_dual_zone_integration.py -v
```
Expected: 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_dual_zone_integration.py
git commit -m "test: add dual-zone chunking integration tests"
```

---

### Task 7: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS (no regressions)

- [ ] **Step 2: Fix any failures, then final commit if changes needed**

```bash
git add -A
git commit -m "chore: final adjustments after full test suite run"
```
