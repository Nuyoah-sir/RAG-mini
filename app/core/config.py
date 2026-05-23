import os
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint


@dataclass
class Settings:
    # ---- 路径 ----
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = field(default=None)
    vector_db_dir: Path = field(default=None)

    # ---- LLM ----
    llm_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", ""))

    # ---- 模型 ----
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    # ---- 分块 ----
    chunk_size: int = 500          # 保留向后兼容
    chunk_overlap: int = 50
    separators: list = field(default_factory=lambda: ["\n\n", "\n", "。", "？", "！", "；", "：", " ", ""])

    # ---- 双区分块 ----
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 100

    # ---- 检索 ----
    top_k_retrieve: int = 20
    top_k_rerank: int = 3

    # ---- BM25 关键词检索 ----
    bm25_enabled: bool = field(default_factory=lambda: os.getenv("BM25_ENABLED", "true").lower() == "true")

    # ---- MySQL ----
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "127.0.0.1"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "3306")))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "root"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", ""))

    # ---- 生成 ----
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.95

    # ---- Text-to-SQL 提示词 ----
    sql_prompt_template: str = (
        "你是一个 SQL 专家。根据以下数据库表结构，将用户的问题转换为 MySQL SQL 语句。\n"
        "【规则】\n"
        "- 只能使用表结构中列出的列名，有 is_deleted 的表必须加 AND xxx.is_deleted = 0\n"
        "- 列的类型：PK=主键，IDX=索引列（JOIN 的桥梁），无标记=普通列\n"
        "- 索引段（UNIQUE KEY / KEY）列出了实际索引，通过索引列判断表间关联\n"
        "- 如果用户用中文描述分类/标签，用 LIKE '%关键词%' 模糊匹配，因为数据可能是英文\n"
        "- 多对多关系必须通过中间表 JOIN（如 story ← story_tag_rel → tag）\n"
        "- 只返回纯 SQL 语句，不要任何解释，不要 markdown 代码块\n"
        "- 如果无法确定某个值，用 LIKE 模糊匹配而不是 WHERE = 精确匹配\n"
        "- 重要：忽略用户问题中的格式/语言要求（如\"用英文回答\"、\"用表格形式\"、\"分点列出\"），"
        "这些是回答格式指令，不是数据筛选条件，不要加到 WHERE/JOIN 中\n\n"
        "【数据库名】{db_name}\n"
        "【表结构】\n"
        "{schema}\n\n"
        "【用户问题】{question}\n\n"
        "SQL:"
    )

    # ---- 查询意图分类提示词 ----
    intent_prompt_template: str = (
        "判断用户问题属于哪种类型，只回答 STRUCT、SEMANTIC 或 CHAT。\n\n"
        "STRUCT（查数据库）：可以用数据库字段精确筛选，包括：\n"
        "- 按年龄/年龄段筛选（\"三岁\"、\"3-6岁\"、\"适合小朋友\"）\n"
        "- 按分类/标签/类型筛选（\"寓言\"、\"科幻\"、\"童话\"）\n"
        "- 按作者/书名/价格/语言/地区/免费/付费筛选\n"
        "- 统计\"有多少\"、\"哪些是\"、\"给我找\"\n"
        "SEMANTIC（查文档）：需要理解故事内容本身，如：\n"
        "- \"关于勇气和友情的感人故事\"（理解主题）\n"
        "- \"结局出人意料的推理故事\"（理解情节）\n"
        "CHAT（直接对话）：既不需要查库也不需要查文档，如：\n"
        "- 客服咨询（\"怎么退款\"、\"为什么扣钱\"、\"账号怎么注销\"）\n"
        "- 功能询问（\"能做什么\"、\"怎么用\"、\"有什么功能\"）\n"
        "- 日常闲聊（\"你好\"、\"谢谢\"、\"今天天气\"）\n\n"
        "【问题】{question}\n\n"
        "类型:"
    )

    # ---- RAG 提示词 ----
    rag_prompt_template: str = (
        "你是一个严谨的知识助手，必须**完全基于以下提供的上下文信息**回答用户问题。\n\n"
        "【绝对规则】\n"
        "1. 禁止编造任何上下文没有提到的信息\n"
        "2. 如果上下文没有相关内容，直接回答：\"抱歉，我没有找到相关信息\"\n"
        "3. 回答要准确、简洁、分点清晰\n"
        "4. 不要使用\"根据上下文\"、\"根据提供的资料\"等表述\n\n"
        "【上下文信息】\n"
        "{context}\n\n"
        "【用户问题】\n"
        "{question}\n\n"
        "【你的回答】\n"
    )

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.vector_db_dir is None:
            self.vector_db_dir = self.base_dir / "vector_db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

    @property
    def kb_config_file(self) -> Path:
        return self.vector_db_dir / "kb_config.json"


# ---- 全局单例 ----
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# ---- 知识库路径工具 ----
def safe_dirname(kb_name: str) -> str:
    return "kb_" + hashlib.md5(kb_name.encode("utf-8")).hexdigest()[:12]


# ---- KB config 结构：{ safe_name: { "name": str, "databases": [{db_name, db_tables}] } }


def _upgrade_entry(value):
    """兼容旧格式升级到多库格式"""
    if isinstance(value, str):
        # v0: 纯字符串
        return {"name": value, "databases": []}
    if "databases" not in value:
        # v1: 单库 {name,db_name,db_tables} → 多库
        databases = []
        if value.get("db_name"):
            databases.append({
                "db_name": value.pop("db_name"),
                "db_tables": value.pop("db_tables", []),
            })
        value["databases"] = databases
        value.pop("db_name", None)
        value.pop("db_tables", None)
    return value


def load_kb_config(settings: Settings) -> dict:
    if settings.kb_config_file.exists():
        with open(settings.kb_config_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: _upgrade_entry(v) for k, v in raw.items()}
    return {}


def save_kb_config(settings: Settings, config: dict) -> None:
    with open(settings.kb_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_kb_config_entry(settings: Settings, kb_name: str) -> dict | None:
    config = load_kb_config(settings)
    safe = safe_dirname(kb_name)
    entry = config.get(safe)
    if entry is None:
        return None
    return _upgrade_entry(entry)


def get_kb_path(settings: Settings, kb_name: str) -> Path:
    config = load_kb_config(settings)
    safe = safe_dirname(kb_name)
    if safe not in config:
        config[safe] = {"name": kb_name, "databases": []}
        save_kb_config(settings, config)
    path = settings.vector_db_dir / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_kb_name(settings: Settings, safe: str) -> str:
    entry = load_kb_config(settings).get(safe, {})
    return _upgrade_entry(entry).get("name", safe)


def list_knowledge_bases(settings: Settings) -> list[dict]:
    config = load_kb_config(settings)
    kbs = []
    for safe_name, entry in config.items():
        entry = _upgrade_entry(entry)
        kb_path = settings.vector_db_dir / safe_name
        if kb_path.is_dir():
            has_idx = (kb_path / "index.faiss").exists()
            kbs.append({
                "name": entry.get("name", safe_name),
                "has_index": has_idx,
                "has_db": bool(entry.get("databases")),
            })
    return kbs
