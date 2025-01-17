import datetime
import logging
import time

import click
from celery import shared_task
from werkzeug.exceptions import NotFound

from core.indexing_runner import IndexingRunner, DocumentIsPausedException
from extensions.ext_database import db
from models.dataset import Document


@shared_task(queue='dataset')
def document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    Usage: document_indexing_task.delay(dataset_id, document_id)
    """
    documents = []
    start_at = time.perf_counter()
    for document_id in document_ids:
        logging.info(click.style(f'Start process document: {document_id}', fg='green'))

        if (
            document := db.session.query(Document)
            .filter(
                Document.id == document_id, Document.dataset_id == dataset_id
            )
            .first()
        ):
            document.indexing_status = 'parsing'
            document.processing_started_at = datetime.datetime.utcnow()
            documents.append(document)
            db.session.add(document)
    db.session.commit()

    try:
        indexing_runner = IndexingRunner()
        indexing_runner.run(documents)
        end_at = time.perf_counter()
        logging.info(
            click.style(
                f'Processed dataset: {dataset_id} latency: {end_at - start_at}',
                fg='green',
            )
        )
    except DocumentIsPausedException as ex:
        logging.info(click.style(str(ex), fg='yellow'))
    except Exception:
        pass
