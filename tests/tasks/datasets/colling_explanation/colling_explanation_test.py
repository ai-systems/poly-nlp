from unittest import TestCase

from dynaconf import settings

from nlp_tasks.tasks.datasets.colling_explanation_dataset.extraction_tasks import (
    CollingExplanationExtractionTask,
)


class TabMCQTest(TestCase):
    def test_worldtree_v1(self):
        dataset = CollingExplanationExtractionTask().run(
            "data/COLING2016_Explanations_Oct2016/Elementary-NDMC-Train-WithExplanations-SubsetWithRelationAnnotation.csv",
        )
        print(dataset)
