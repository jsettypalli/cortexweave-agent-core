from sqlalchemy.orm import Session

from cortexweave_core.rag.pipelines.documents.models import KnowledgeBase


class KnowledgeBaseDAO:
    @staticmethod
    def find_by_name(session: Session, name: str) -> KnowledgeBase | None:
        return session.query(KnowledgeBase).filter_by(name=name).first()
