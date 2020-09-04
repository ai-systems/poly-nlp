from unittest import TestCase

from poly_nlp.utils.result_handler import PolyNLPCaching, PolyNLPResultHandler


class PolyResultHandlerTest(TestCase):
    def test_msgpack_caching(self):
        result_handler = PolyNLPResultHandler()
        ...
