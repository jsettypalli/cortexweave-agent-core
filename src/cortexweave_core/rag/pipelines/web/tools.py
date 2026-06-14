from cortexweave_core.rag.pipelines.documents.tools import search_documents


async def search_web_content(query: str) -> dict:
    """
    Performs a keyword search over web content that has been crawled and
    ingested into the configured knowledge base / collection, returning the
    most relevant matches.

    Crawled web pages are stored as documents in the same collection, so this
    delegates directly to the documents pipeline's search.

    Args:
        query: Free-text search query. Individual words are matched against
            page content (case-insensitive).

    Returns:
        dict with "results": a list of {"content", "source", "metadata"} for
        each matching page (up to 10). If no knowledge base/collection is
        configured or no content is found, "results" is empty and an "error"
        key explains why.
    """
    return await search_documents(query)
