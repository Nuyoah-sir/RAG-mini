"""
RAG REST API 服务 — 供 Java Web / 其他项目调用
启动: python api.py
端口: 7880
"""
import os
import json
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch

from config import *
from app import (
    get_embedding_model, get_reranker, get_llm,
    get_or_load_kb, kb_stores, kb_chains,
    load_documents, split_documents, build_vector_store,
    create_qa_chain, list_knowledge_bases, get_kb_path,
)

# ---- 请求/响应模型 ----
class AskRequest(BaseModel):
    kb_name: str
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    kb_name: str

class KBCreateRequest(BaseModel):
    kb_name: str

class KBBuildRequest(BaseModel):
    kb_name: str
    file_paths: list[str]  # 服务端文件路径列表

# ---- 应用生命周期 ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热模型
    get_embedding_model()
    get_reranker()
    get_llm()
    # 预加载已有知识库
    for kb in list_knowledge_bases():
        if kb["has_index"]:
            get_or_load_kb(kb["name"])
    print("🚀 RAG API 服务已就绪")
    yield

app = FastAPI(title="多知识库 RAG API", lifespan=lifespan)

# 允许跨域（Java Web 前端调用需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- API 端点 ----
@app.get("/api/kb/list")
def api_list_kbs():
    """列出所有知识库"""
    return {"knowledge_bases": list_knowledge_bases()}


@app.post("/api/kb/create")
def api_create_kb(req: KBCreateRequest):
    """创建新知识库"""
    if not req.kb_name or not req.kb_name.strip():
        raise HTTPException(400, "知识库名称不能为空")
    kb_path = get_kb_path(req.kb_name.strip())
    os.makedirs(kb_path, exist_ok=True)
    return {"status": "ok", "kb_name": req.kb_name.strip()}


@app.post("/api/kb/{kb_name}/build")
def api_build_kb(kb_name: str, req: KBBuildRequest):
    """构建知识库（从服务端已有文件）"""
    if not req.file_paths:
        raise HTTPException(400, "文件路径列表不能为空")

    documents = load_documents(req.file_paths)
    if not documents:
        raise HTTPException(400, "没有成功加载任何文档")

    chunks = split_documents(documents)
    vs = build_vector_store(chunks, kb_name)
    kb_stores[kb_name] = vs
    kb_chains[kb_name] = create_qa_chain(vs, kb_name)

    return {
        "status": "ok",
        "kb_name": kb_name,
        "documents": len(documents),
        "chunks": len(chunks)
    }


@app.post("/api/kb/{kb_name}/build-from-files")
async def api_build_kb_from_files(kb_name: str):
    """构建知识库（从 data 目录读取文件 — 先用 upload 端点上传文件）"""
    raise HTTPException(501, "请先通过文件上传接口上传文档到 data/ 目录")


@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    """问答接口 — Java 等外部系统调用的核心端点"""
    chain = get_or_load_kb(req.kb_name)
    if not chain:
        raise HTTPException(404, f"知识库 [{req.kb_name}] 不存在或为空")

    try:
        result = chain.invoke({"query": req.question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        sources = []
        for i, doc in enumerate(source_docs):
            src = {"index": i + 1}
            if "source" in doc.metadata:
                src["file"] = os.path.basename(doc.metadata["source"])
            if "page" in doc.metadata:
                src["page"] = doc.metadata["page"] + 1
            src["snippet"] = doc.page_content[:200]
            sources.append(src)

        return AskResponse(
            answer=answer,
            sources=sources,
            kb_name=req.kb_name
        )
    except Exception as e:
        raise HTTPException(500, f"问答失败: {str(e)}")


@app.delete("/api/kb/{kb_name}")
def api_delete_kb(kb_name: str):
    """删除知识库"""
    kb_path = get_kb_path(kb_name)
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)
    kb_stores.pop(kb_name, None)
    kb_chains.pop(kb_name, None)
    return {"status": "deleted", "kb_name": kb_name}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---- 启动 ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7880)
