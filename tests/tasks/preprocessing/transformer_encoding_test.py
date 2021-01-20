from unittest import TestCase

from poly_nlp.tasks.preprocessing.transformer_encode_task import TransformerEncodeTask


class EncodingTextTest(TestCase):
    def test_text_encoding(self):
        text_encoding = TransformerEncodeTask()
        output = text_encoding.run(
            text_input={"1": "I am running this code"},
            output_path="data/checkpoint",
            task_name="test",
            transformer_model="bert-base-uncased",
        )
        print(output)
        for id, val in output.items():
            print(id, val)
        print(output["1"])
