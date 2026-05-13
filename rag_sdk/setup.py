from setuptools import setup, find_packages

setup(
    name="rag-sdk",
    version="1.0.0",
    description="多知识库 RAG 检索引擎 — 支持私有部署 LLM + DeepSeek 重排序",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.3.0,<0.4.0",
        "langchain-community>=0.3.0,<0.4.0",
        "langchain-openai>=0.3.5,<0.4.0",
        "langchain-huggingface<1.0",
        "langchain-text-splitters",
        "sentence-transformers>=3.0",
        "faiss-cpu>=1.8.0",
        "pypdf>=4.0",
        "docx2txt>=0.9",
        "python-docx>=1.1",
        "python-dotenv>=1.0",
        "requests>=2.0",
        "torch",
    ],
)
