from unittest import TestCase

from dynaconf import settings

from poly_nlp.tasks.datasets.tab_mcq.extraction_tasks import (
    AristoTableStoreExtractionTask,
)


class TabMCQTest(TestCase):
    def test_worldtree_v1(self):
        table_store_path = settings["worldtree_v1"]["table_store"]
        table_store = AristoTableStoreExtractionTask().run(
            "data/TabMCQ/Tables/monarch", "monarch"
        )
        print(len(table_store))
