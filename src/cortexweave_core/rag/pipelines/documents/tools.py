import asyncio

from cortexweave_core.rag.db import get_session
from cortexweave_core.rag.pipelines.documents.dao import DocumentDAO, KnowledgeBaseDAO, KnowledgeCollectionDAO
from cortexweave_core.utils.config_loader import config


def _config_or_raise(key: str) -> str:
    value = config.get(key)
    if not value:
        raise RuntimeError(f"{key} is not configured for this sub-agent")
    return value


def _not_found(knowledge_base_name: str, collection_name: str) -> dict:
    return {
        "results": [],
        "error": f"No documents found for collection '{collection_name}' in knowledge base '{knowledge_base_name}'.",
    }


def _search(query: str, limit: int = 10) -> dict:
    knowledge_base_name = _config_or_raise("RAG_KNOWLEDGE_BASE_NAME")
    collection_name = _config_or_raise("RAG_COLLECTION_NAME")
    session = get_session()
    try:
        kb = KnowledgeBaseDAO.find_by_name(session, knowledge_base_name)
        if not kb:
            return _not_found(knowledge_base_name, collection_name)
        collection = KnowledgeCollectionDAO.find_by_name(session, collection_name, kb.id)
        if not collection:
            return _not_found(knowledge_base_name, collection_name)
        terms = [t for t in query.split() if t]
        documents = DocumentDAO.find_keyword_matches(session, collection.id, terms, limit=limit)
        return {"results": [
            {"content": d.content, "source": d.source, "metadata": d.metadata_json or {}}
            for d in documents
        ]}
    finally:
        session.close()


async def search_documents(query: str) -> dict:
    """
    Performs a keyword search over the configured knowledge base / collection's
    ingested documents and returns the most relevant matches.

    Args:
        query: Free-text search query. Individual words are matched against
            document content (case-insensitive).

    Returns:
        dict with "results": a list of {"content", "source", "metadata"} for
        each matching document (up to 10). If no knowledge base/collection is
        configured or no documents are found, "results" is empty and an
        "error" key explains why.
    """
    return await asyncio.to_thread(_search, query)
