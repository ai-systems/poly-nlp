from unittest import TestCase

from dynaconf import settings


class QASCExtractionTest(TestCase):
    def test_qasc_extraction(self):
        table_store_path = settings["worldtree_v1"]["table_store"]
