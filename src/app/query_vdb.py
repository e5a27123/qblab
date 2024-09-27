import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

load_dotenv(override=True)


class ChromaDBClient:
    def __init__(self, collection_name: str):
        """
        初始化 ChromaDBClient 類別，建立與 Chroma DB 的連接並創建或取得指定的集合。

        :param collection_name: Chroma DB 集合的名稱
        """
        self.collection_name = collection_name
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()
        self.vectorstore = self._get_or_create_vectorstore()

    def _initialize_client(self) -> chromadb.HttpClient:
        """
        初始化 Chroma DB HTTP 客戶端。

        :return: 初始化的 Chroma DB 客戶端
        """
        return chromadb.HttpClient(
            host=os.environ["CHROMA_HOST"],
            port=os.environ["CHROMA_PORT"],
        )

    def _get_azure_openai_ef(self) -> embedding_functions.OpenAIEmbeddingFunction:
        """
        取得 Azure OpenAI embedding function。

        :return: Azure OpenAI embedding function
        """
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
            model_name="text-embedding-3-small",
            api_base=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
            api_type="azure",
            api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
            deployment_id=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        )
        return openai_ef

    def _get_or_create_collection(self):
        """
        創建或取得指定的集合。

        :return: Chroma DB 集合物件
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._get_azure_openai_ef(),
        )

    def query(self, query_texts: str, n_results: int = 1):
        """
        查詢 Chroma DB 集合。

        :param query_texts: 查詢的文本
        :param n_results: 要返回的結果數量
        :return: 查詢結果
        """
        return self.collection.query(query_texts=query_texts, n_results=n_results)

    def _get_or_create_vectorstore(
        self,
        collection_name: str = "collect01",
    ) -> Chroma:
        vectordb = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self._get_Embeddings_func(),
        )
        return vectordb

    def _get_Embeddings_func(self) -> AzureOpenAIEmbeddings:
        """
        Get the embeddings function based on the specified type.

        :param emb_type: Type of embeddings to use ("azure" or other).
        :return: Embeddings function.
        """
        embeddings = AzureOpenAIEmbeddings(
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        )
        return embeddings


# # 使用範例
if __name__ == "__main__":
        collection_name = "collect_cubelab_qa_lite"
        chroma_client = ChromaDBClient(collection_name=collection_name)
        result = chroma_client.query(query_texts="test123", n_results=1)
        print(result)

        result = chroma_client.vectorstore.similarity_search_with_relevance_scores(
            query="test123", k=1, score_threshold=0
        )
        print(result)
    # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
    # from langchain_core.prompts import ChatPromptTemplate

    # embeddings = AzureOpenAIEmbeddings(
    #     api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
    #     azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
    #     # api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     # azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    #     azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    #     api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    # )
    # # print(embeddings.embed_query("hi"))

    # model = AzureChatOpenAI(
    #     api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    #     openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    #     azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    # )

    # chain = ChatPromptTemplate.from_template("{input}") | model
    # print(chain.invoke({"input": "hi"}))
