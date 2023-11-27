import logging
from typing import Optional, List

import cohere
import openai
from langchain.schema import Document

from core.model_providers.error import LLMBadRequestError, LLMAPIConnectionError, LLMAPIUnavailableError, \
    LLMRateLimitError, LLMAuthorizationError
from core.model_providers.models.reranking.base import BaseReranking
from core.model_providers.providers.base import BaseModelProvider


class CohereReranking(BaseReranking):

    def __init__(self, model_provider: BaseModelProvider, name: str):
        self.credentials = model_provider.get_model_credentials(
            model_name=name,
            model_type=self.type
        )

        client = cohere.Client(self.credentials.get('api_key'))

        super().__init__(model_provider, client, name)

    def rerank(self, query: str, documents: List[Document], score_threshold: Optional[float], top_k: Optional[int]) -> Optional[List[Document]]:
        docs = []
        doc_id = []
        for document in documents:
            if document.metadata['doc_id'] not in doc_id:
                doc_id.append(document.metadata['doc_id'])
                docs.append(document.page_content)
        results = self.client.rerank(query=query, documents=docs, model=self.name, top_n=top_k)
        rerank_documents = []

        for result in results:
            # format document
            rerank_document = Document(
                page_content=result.document['text'],
                metadata={
                    "doc_id": documents[result.index].metadata['doc_id'],
                    "doc_hash": documents[result.index].metadata['doc_hash'],
                    "document_id": documents[result.index].metadata['document_id'],
                    "dataset_id": documents[result.index].metadata['dataset_id'],
                    'score': result.relevance_score
                }
            )
            # score threshold check
            if (
                score_threshold is not None
                and result.relevance_score >= score_threshold
                or score_threshold is None
            ):
                rerank_documents.append(rerank_document)
        return rerank_documents

    def handle_exceptions(self, ex: Exception) -> Exception:
        if isinstance(ex, openai.error.InvalidRequestError):
            logging.warning("Invalid request to OpenAI API.")
            return LLMBadRequestError(str(ex))
        elif isinstance(ex, openai.error.APIConnectionError):
            logging.warning("Failed to connect to OpenAI API.")
            return LLMAPIConnectionError(f"{ex.__class__.__name__}:{str(ex)}")
        elif isinstance(ex, (openai.error.APIError, openai.error.ServiceUnavailableError, openai.error.Timeout)):
            logging.warning("OpenAI service unavailable.")
            return LLMAPIUnavailableError(f"{ex.__class__.__name__}:{str(ex)}")
        elif isinstance(ex, openai.error.RateLimitError):
            return LLMRateLimitError(str(ex))
        elif isinstance(ex, openai.error.AuthenticationError):
            return LLMAuthorizationError(str(ex))
        elif isinstance(ex, openai.error.OpenAIError):
            return LLMBadRequestError(f"{ex.__class__.__name__}:{str(ex)}")
        else:
            return ex
