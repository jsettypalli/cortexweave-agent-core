from sqlalchemy import Boolean, Column, ForeignKey, Integer, JSON, String, Text

from cortexweave_core.rag.db import RAGReadBase


class KnowledgeBase(RAGReadBase):
    __tablename__ = "rag_knowledge_bases"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)


class KnowledgeCollection(RAGReadBase):
    __tablename__ = "rag_collections"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    knowledge_base_id = Column(Integer, ForeignKey("rag_knowledge_bases.id"), nullable=False)


class Document(RAGReadBase):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    collection_id = Column(Integer, ForeignKey("rag_collections.id"), nullable=False)
    source = Column(String(255), nullable=True)
    metadata_json = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
