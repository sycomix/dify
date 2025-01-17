import logging
import time

import click
from celery import shared_task
from werkzeug.exceptions import NotFound

from core.index.index import IndexBuilder
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import DocumentSegment, Dataset, Document


@shared_task(queue='dataset')
def delete_segment_from_index_task(segment_id: str, index_node_id: str, dataset_id: str, document_id: str):
    """
    Async Remove segment from index
    :param segment_id:
    :param index_node_id:
    :param dataset_id:
    :param document_id:

    Usage: delete_segment_from_index_task.delay(segment_id)
    """
    logging.info(
        click.style(
            f'Start delete segment from index: {segment_id}', fg='green'
        )
    )
    start_at = time.perf_counter()
    indexing_cache_key = f'segment_{segment_id}_delete_indexing'
    try:
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            logging.info(
                click.style(
                    f'Segment {segment_id} has no dataset, pass.', fg='cyan'
                )
            )
            return

        dataset_document = db.session.query(Document).filter(Document.id == document_id).first()
        if not dataset_document:
            logging.info(
                click.style(
                    f'Segment {segment_id} has no document, pass.', fg='cyan'
                )
            )
            return

        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != 'completed':
            logging.info(
                click.style(
                    f'Segment {segment_id} document status is invalid, pass.',
                    fg='cyan',
                )
            )
            return

        vector_index = IndexBuilder.get_index(dataset, 'high_quality')
        kw_index = IndexBuilder.get_index(dataset, 'economy')

        # delete from vector index
        if vector_index:
            vector_index.delete_by_ids([index_node_id])

        # delete from keyword index
        kw_index.delete_by_ids([index_node_id])

        end_at = time.perf_counter()
        logging.info(
            click.style(
                f'Segment deleted from index: {segment_id} latency: {end_at - start_at}',
                fg='green',
            )
        )
    except Exception:
        logging.exception("delete segment from index failed")
    finally:
        redis_client.delete(indexing_cache_key)
