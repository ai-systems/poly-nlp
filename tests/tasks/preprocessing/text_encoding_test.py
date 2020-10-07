from unittest import TestCase

from poly_nlp.tasks.preprocessing.encode_text import EncodeTextTask


class EncodingTextTest(TestCase):
    def test_text_encoding(self):
        glove_embedding_file = "/home/mohan/Projects/../glove.6B.100d.txt"
        text_encoding = EncodeTextTask()
        output = text_encoding.run(
            text_input={"1": "I am running this code"},
            pretrained_file=glove_embedding_file,
            output_path="data/checkpoint",
        )
        for id, val in output["inputs"].items():
            print(id, val)
        print(output["inputs"]["1"])

