from unittest import TestCase
from poly_nlp.tasks.preprocessing.question_to_hypothesis import QuestionHypothesisExtractionTask
from dynaconf import settings


class QuestionHypothesisTest(TestCase):
    def quetion_hypothesis_test(self):
        instance = {'1':{'question':'Which features can be found on the surface of both Earth and the Moon?', 'answer':'mountains'}}
        models_dir=settings['stanford_model']
        output = QuestionHypothesisExtractionTask(models_dir).run(instance)
        print(output)
       


