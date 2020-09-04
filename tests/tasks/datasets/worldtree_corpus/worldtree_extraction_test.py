from unittest import TestCase

from dynaconf import settings

from poly_nlp.tasks.datasets.worldtree.extraction_tasks import (
    TableStoreExtractionTask,
    WorldTreeExtractionTask,
    WorldTreeVersion,
)


class ExplanationBankTest(TestCase):
    def test_worldtree_v1(self):
        table_store_path = settings["worldtree_v1"]["table_store"]
        expl_bank_data_files = settings["worldtree_v1"]["dev"]
        table_store = TableStoreExtractionTask().run(table_store_path)
        explanation_bank = WorldTreeExtractionTask().run(
            expl_bank_data_files, WorldTreeVersion.WorldTree_V1
        )

    def test_worldtree_v2(self):
        table_store_path = settings["worldtree_v2"]["table_store"]
        expl_bank_data_files = settings["worldtree_v2"]["dev"]
        table_store = TableStoreExtractionTask().run(table_store_path)
        explanation_bank = WorldTreeExtractionTask().run(
            expl_bank_data_files, WorldTreeVersion.WorldTree_V2
        )
        assert explanation_bank["MDSA_2009_5_16_ENUM0"]["fold"] == "Challenge"
        assert (
            explanation_bank["MDSA_2009_5_16_ENUM0"]["question"]
            == "Students visited the Morris W. Offit telescope located at the Maryland Space Grant Observatory in Baltimore. They learned about the stars, planets, and moon. The students recorded the information below. • Star patterns stay the same, but their locations in the sky seem to change. • The sun, planets, and moon appear to move in the sky. • Proxima Centauri is the nearest star to our solar system. • Polaris is a star that is part of a pattern of stars called the Little Dipper. Which statement best explains why the sun appears to move across the sky each day?"
        )
        assert (
            explanation_bank["MDSA_2009_5_16_ENUM0"]["answer"]
            == "Earth rotates on its axis."
        )
