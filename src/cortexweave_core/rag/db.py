from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from cortexweave_core.utils.config_loader import config


class RAGReadBase(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def _get_engine():
    db_url = config.get("RAG_DATABASE_URL")
    if not db_url:
        raise RuntimeError("RAG_DATABASE_URL is not configured")
    return create_engine(db_url, pool_pre_ping=True)


def get_session() -> Session:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return SessionLocal()
