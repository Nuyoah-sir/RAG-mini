# Dual-Zone Chunking Design

**Date:** 2026-05-23
**Status:** approved

## 1. 问题

当前 `RecursiveCharacterTextSplitter` 对所有文档使用相同的 chunk_size=500、chunk_overlap=50，所有 chunk 平铺建索引。chunk 之间没有关联，检索返回的 3 个 chunk 可能来自同一文档但缺乏叙事连贯性——对儿童故事场景尤其有害。

## 2. 方案：父子双区分块

**子块索引用、父块生成用。** 子块小（~400字），确保语义匹配精准；父块大（~1200字），保留叙事完整性。检索命中子块后，通过 parent_id 映射返回对应父块给 LLM。

## 3. 架构

```
文档加载
  │
  ▼
DualZoneSplitter
  ├─ 父块: chunk_size=1200, overlap=200
  └─ 子块: chunk_size=400, overlap=100  ← 在父块范围内再切
  │
  ▼
build_vector_store()
  ├─ 子块 → FAISS 索引（检索）
  ├─ 子块 → BM25 索引（检索）
  └─ parent_map → pickle 持久化
  │
  ▼
检索时: query → FAISS/BM25 命中子块
       → parent_map 查找父块
       → parent_id 去重
       → 父块 → reranker → LLM
```

## 4. 改动文件

| 文件 | 改动 |
|------|------|
| `app/component/splitter.py` | 重写：DualZoneSplitter，父块先切→子块在父块内再切，子块 metadata 注入 parent_id |
| `app/core/config.py` | 新增 4 个配置项 |
| `app/service/document_service.py` | build_kb() 接收并传递 parent_map |
| `app/repository/vector_repo.py` | build_vector_store() 保存 parent_map.pkl；load_vector_store() 加载 parent_map |
| `app/service/rag_service.py` | HybridRetriever 增加 parent_map 属性，检索末尾做子块→父块转换+去重 |

## 5. 配置新增

```python
# ---- 双区分块 ----
parent_chunk_size: int = 1200
parent_chunk_overlap: int = 200
child_chunk_size: int = 400
child_chunk_overlap: int = 100
```

保留原有 `chunk_size`/`chunk_overlap` 用于向后兼容（无 parent_map 的旧索引回退到原始行为）。

## 6. 数据结构

```python
# parent_map: dict[str, Document]
#   key = parent_id (UUID 字符串)
#   value = 父块 Document（page_content + metadata）
#
# 子块 metadata:
#   doc.metadata["parent_id"] = parent_id  # 指向父块
```

## 7. 数据流

### 建索引
```
load_documents(files) → docs
DualZoneSplitter.split_documents(docs) → (child_chunks, parent_map)
build_vector_store(child_chunks, parent_map, kb_name)
  → FAISS.from_documents(child_chunks)
  → BM25.build(child_chunks)
  → pickle.dump(parent_map, "parent_map.pkl")
```

### 检索
```
query → HybridRetriever._get_relevant_documents()
  → FAISS + BM25 各自检索 → 合并去重（子块）
  → 遍历子块，parent_map.get(parent_id) → 父块
  → parent_id 去重（多子块命中同父块只保留一份）
  → 返回父块列表
  → ContextualCompressionRetriever(reranker) 压缩
  → top 3 父块 → LLM
```

## 8. 向后兼容

- 旧索引（无 parent_map.pkl）：检索时检测 parent_map 为空，直接返回子块（回退原有行为）
- 旧配置无 4 个新字段：Settings dataclass 有默认值，无需迁移

## 9. 边界情况

- **单 chunk 文档**：子块=父块，parent_map 只有一条映射
- **父块不足子块大小**：_child_splitter 对小于 chunk_size 的内容不切分，直接作为唯一子块
- **多子块命中同父块**：parent_id 去重，避免 LLM 收到重复上下文
- **BM25 disabled**：不影响，parent_map 独立于 BM25
