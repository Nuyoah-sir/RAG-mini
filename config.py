import os
from dotenv import load_dotenv

load_dotenv()

# 基础目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# 自动创建必要目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# 大模型API配置（从.env读取）
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# DeepSeek重排序API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

# 本地嵌入模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 文本分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "？", "！", "；", "：", " ", ""]

# 检索参数
TOP_K_RETRIEVE = 20  # 初步召回数量
TOP_K_RERANK = 3     # 重排序后保留数量

# LLM生成参数
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.95

# RAG提示模板（针对Gemma模型优化）
RAG_PROMPT_TEMPLATE = """
你是一个严谨的知识助手，必须**完全基于以下提供的上下文信息**回答用户问题。

【绝对规则】
1. 禁止编造任何上下文没有提到的信息
2. 如果上下文没有相关内容，直接回答："抱歉，我没有找到相关信息"
3. 回答要准确、简洁、分点清晰
4. 不要使用"根据上下文"、"根据提供的资料"等表述

【上下文信息】
{context}

【用户问题】
{question}

【你的回答】
"""


import hashlib
import json

KB_CONFIG_FILE = os.path.join(VECTOR_DB_DIR, "kb_config.json")


def _safe_dirname(kb_name):
    """用 MD5 生成安全的目录名（FAISS 不支持中文路径）"""
    return "kb_" + hashlib.md5(kb_name.encode("utf-8")).hexdigest()[:12]


def _load_kb_config():
    """加载知识库配置: {safe_dirname: display_name}"""
    if os.path.exists(KB_CONFIG_FILE):
        with open(KB_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_kb_config(config):
    with open(KB_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_kb_path(kb_name):
    """获取指定知识库的向量存储路径（内部使用安全目录名）"""
    config = _load_kb_config()
    safe_name = _safe_dirname(kb_name)
    # 注册映射
    if safe_name not in config:
        config[safe_name] = kb_name
        _save_kb_config(config)
    path = os.path.join(VECTOR_DB_DIR, safe_name)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_kb_name(safe_dirname):
    """根据安全目录名反查显示名"""
    config = _load_kb_config()
    return config.get(safe_dirname, safe_dirname)


def list_knowledge_bases():
    """列出所有已创建的知识库"""
    config = _load_kb_config()
    kbs = []
    for safe_name, display_name in config.items():
        kb_path = os.path.join(VECTOR_DB_DIR, safe_name)
        if os.path.isdir(kb_path):
            index_file = os.path.join(kb_path, "index.faiss")
            kbs.append({"name": display_name, "has_index": os.path.exists(index_file)})
    return kbs
