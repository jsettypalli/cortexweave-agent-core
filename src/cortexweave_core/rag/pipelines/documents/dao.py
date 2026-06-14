from sqlalchemy import or_
from sqlalchemy.orm import Session

from cortexweave_core.rag.pipelines.documents.models import Document, KnowledgeBase, KnowledgeCollection


class KnowledgeBaseDAO:
    @staticmethod
    def find_by_name(session: Session, name: str) -> KnowledgeBase | None:
        return session.query(KnowledgeBase).filter_by(name=name).first()


class KnowledgeCollectionDAO:
    @staticmethod
    def find_by_name(session: Session, name: str, knowledge_base_id: int) -> KnowledgeCollection | None:
        return session.query(KnowledgeCollection).filter_by(
            name=name, knowledge_base_id=knowledge_base_id
        ).first()


class DocumentDAO:
    @staticmethod
    def find_keyword_matches(
        session: Session, collection_id: int, terms: list[str], limit: int = 10
    ) -> list[Document]:
        terms = [t.strip() for t in terms if t and t.strip()]
        if not terms:
            return []
        query = session.query(Document).filter(
            Document.collection_id == collection_id, Document.is_active,
        ).filter(or_(*[Document.content.ilike(f"%{term}%") for term in terms]))
        return query.limit(limit).all()
