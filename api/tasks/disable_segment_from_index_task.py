import logging
import time

import click
from celery import shared_task
from werkzeug.exceptions import NotFound

from core.index.index import IndexBuilder
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import DocumentSegment


@shared_task(queue='dataset')
def disable_segment_from_index_task(segment_id: str):
    """
    Async disable segment from index
    :param segment_id:

    Usage: disable_segment_from_index_task.delay(segment_id)
    """
    logging.info(
        click.style(
            f'Start disable segment from index: {segment_id}', fg='green'
        )
    )
    start_at = time.perf_counter()

    segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_id).first()
    if not segment:
        raise NotFound('Segment not found')

    if segment.status != 'completed':
        return

    indexing_cache_key = f'segment_{segment.id}_indexing'

    try:
        dataset = segment.dataset

        if not dataset:
            logging.info(
                click.style(
                    f'Segment {segment.id} has no dataset, pass.', fg='cyan'
                )
            )
            return

        dataset_document = segment.document

        if not dataset_document:
            logging.info(
                click.style(
                    f'Segment {segment.id} has no document, pass.', fg='cyan'
                )
            )
            return

        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != 'completed':
            logging.info(
                click.style(
                    f'Segment {segment.id} document status is invalid, pass.',
                    fg='cyan',
                )
            )
            return

        vector_index = IndexBuilder.get_index(dataset, 'high_quality')
        kw_index = IndexBuilder.get_index(dataset, 'economy')

        # delete from vector index
        if vector_index:
            vector_index.delete_by_ids([segment.index_node_id])

        # delete from keyword index
        kw_index.delete_by_ids([segment.index_node_id])

        end_at = time.perf_counter()
        logging.info(
            click.style(
                f'Segment removed from index: {segment.id} latency: {end_at - start_at}',
                fg='green',
            )
        )
    except Exception:
        logging.exception("remove segment from index failed")
        segment.enabled = True
        db.session.commit()
    finally:
        redis_client.delete(indexing_cache_key)
