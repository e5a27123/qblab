
from collections import Counter
from langchain_core.runnables import RunnableLambda
from typing import List, Callable, Dict, Union, Any
from operator import itemgetter
import re

from langchain_core.documents import Document
from langchain_core.runnables import chain


def _get_metadata(response: List[Any], *keys: str) -> List[Any]:
    """
    Retrieve metadata based on provided keys from the response.

    For single-element responses, return the value directly.
    For multiple-element responses, return the maximum value for numerical types,
    or the most common value for other types.

    Parameters:
    response -- A list containing metadata
    *keys -- The keys to retrieve from the metadata

    Returns:
    A list containing the retrieved values.
    """

    if not response:
        return []

    if len(response) == 1:
        result = itemgetter(*keys)(response[0].metadata)
    else:
        metadata_list = [itemgetter(*keys)(item.metadata) for item in response]

        if isinstance(metadata_list[0], (int, float)):
            result = max(metadata_list)
        else:
            counter = Counter(metadata_list)
            result = counter.most_common(1)[0][0]

    return [result] if len(keys) == 1 else list(result)


def _create_partial_func(func: Callable, *keys: str) -> Callable:
    """
    Creates a partial function that calls the original function
    with the provided keys when invoked.

    Parameters:
    func -- The original function
    *keys -- The keys to pass to the original function

    Returns:
    A partial function that calls the original function
    with the provided keys when invoked.
    """

    def partial_func(response):
        return func(response, *keys)

    return partial_func


def get_metadata_runnable(*keys: str) -> Callable[[List], List]:
    """
    Creates a langchain runnable with the provided keys when invoked.

    Parameters:
    *keys -- The keys to pass to the function

    Returns:
    A langchain runnable function with the provided keys when invoked.
    """
    f = _create_partial_func(_get_metadata, *keys)
    return RunnableLambda(f)


def get_sql_querys(response: Dict[str, Dict[str, str]], user_id: str) -> List[str]:
    """
    Get processed SQL queries based on the response and customer ID.

    Args:
        response (Dict[str, Dict[str, str]]):
            A dictionary containing keys, values, and SQL queries.
        user_id (str):
            The customer ID to replace placeholders in the queries.

    Returns:
        List[str]: List of processed SQL queries.
    """

    keys = response["keys"]
    keys["&CustomerID"] = user_id

    querys = []
    for query in response["retriever"]["SQL"]:
        querys.append(replace_sql_query(query, keys))

    return querys


def RetrieveWithScore(
    vectorstore: "VectorStore",
    k: int = 4,
    score_threshold: float = 0.8,
) -> Callable[[str], List[Document]]:
    """
    Creates a retriever to fetch relevant documents from a vector store.

    Args:
        vectorstore (VectorStore): The vector store instance.
        k (int, optional): Max number of documents to retrieve. Default is 4.
        score_threshold (float, optional): Minimum relevance score. Default is 0.8.

    Returns:
        Callable[[str], List[Document]]: A function that retrieves documents.
    """

    @chain
    def retriever(query: str) -> List[Document]:
        """
        Retrieves relevant documents based on a query.

        Args:
            query (str): The query string.

        Returns:
            List[Document]: List of documents with relevance scores.
        """
        result = vectorstore.similarity_search_with_relevance_scores(
            query=query, k=k, score_threshold=score_threshold
        )
        if result:
            docs, scores = zip(*result)
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score
            return list(docs)
        return []

    return retriever
