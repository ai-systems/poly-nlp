from uuid import uuid4

from overrides import overrides
from prefect import Task


class TextCorpusExtractionTask(Task):
    @overrides
    def run(self, path, filter_fn=None, text_processs_fn=None):
        table_store = {}
        with open(path) as f:
            for line in f:
                skip = False
                fact = line.strip()
                if text_processs_fn is not None:
                    fact = text_processs_fn(fact)
                if filter_fn is not None:
                    skip = filter_fn(fact)
                if not skip:
                    table_store[str(uuid4())] = {"fact": fact}
        return table_store
