import logging
import time

import click
from celery import shared_task
from langchain.schema import Document

from core.index.index import IndexBuilder
from extensions.ext_database import db
from models.dataset import DocumentSegment, Dataset
from models.dataset import Document as DatasetDocument


@shared_task(queue='dataset')
def deal_dataset_vector_index_task(dataset_id: str, action: str):
    """
    Async deal dataset from index
    :param dataset_id: dataset_id
    :param action: action
    Usage: deal_dataset_vector_index_task.delay(dataset_id, action)
    """
    logging.info(
        click.style(
            f'Start deal dataset vector index: {dataset_id}', fg='green'
        )
    )
    start_at = time.perf_counter()

    try:
        dataset = Dataset.query.filter_by(
            id=dataset_id
        ).first()

        if not dataset:
            raise Exception('Dataset not found')

        if action == "remove":
            index = IndexBuilder.get_index(dataset, 'high_quality', ignore_high_quality_check=True)
            index.delete_by_group_id(dataset.id)
        elif action == "add":
            if (
                dataset_documents := db.session.query(DatasetDocument)
                .filter(
                    DatasetDocument.dataset_id == dataset_id,
                    DatasetDocument.indexing_status == 'completed',
                    DatasetDocument.enabled == True,
                    DatasetDocument.archived == False,
                )
                .all()
            ):
                # save vector index
                index = IndexBuilder.get_index(dataset, 'high_quality', ignore_high_quality_check=False)
                documents = []
                for dataset_document in dataset_documents:
                    # delete from vector index
                    segments = db.session.query(DocumentSegment).filter(
                        DocumentSegment.document_id == dataset_document.id,
                        DocumentSegment.enabled == True
                    ) .order_by(DocumentSegment.position.asc()).all()
                    for segment in segments:
                        document = Document(
                            page_content=segment.content,
                            metadata={
                                "doc_id": segment.index_node_id,
                                "doc_hash": segment.index_node_hash,
                                "document_id": segment.document_id,
                                "dataset_id": segment.dataset_id,
                            }
                        )

                        documents.append(document)

                # save vector index
                index.create(documents)

        end_at = time.perf_counter()
        logging.info(
            click.style(
                f'Deal dataset vector index: {dataset_id} latency: {end_at - start_at}',
                fg='green',
            )
        )
    except Exception:
        logging.exception("Deal dataset vector index failed")
