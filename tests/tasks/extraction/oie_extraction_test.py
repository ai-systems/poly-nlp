import unittest

from poly_nlp.tasks.extraction.allen_oie_extraction_task import AllenOIEExtractionTask


class OIEExtractionTest(unittest.TestCase):
    def test_oie_extraction(self):
        sentences = {"1": "In December, John decided to join the party"}
        results = AllenOIEExtractionTask().run(
            sentences,
            "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
        )
        print(results)
