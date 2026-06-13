from sqlalchemy import Column, Integer, String

from cortexweave_core.rag.db import RAGReadBase


class KnowledgeBase(RAGReadBase):
    __tablename__ = "rag_knowledge_bases"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
