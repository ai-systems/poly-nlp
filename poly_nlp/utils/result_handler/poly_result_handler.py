import os
import typing
from enum import Enum
from pathlib import Path
from typing import Any

import cloudpickle
import loguru
import msgpack
import ujson
from loguru import logger
from overrides import overrides
from prefect.engine.result_handlers import result_handler
from prefect.engine.results import LocalResult


class PolyNLPCaching(Enum):
    MSGPACK = msgpack
    CLD_PICKLE = cloudpickle
    JSON = ujson


class PolyNLPLocalResult(result_handler.ResultHandler):
    def __init__(
        self,
        dir: str,
        caching_method: PolyNLPCaching = PolyNLPCaching.MSGPACK,
        read_cache: bool = True,
        **kwargs: Any,
    ):
        super().__init__(dir, **kwargs)
        self.read_cache = read_cache
        self.caching_method = caching_method

     def read(self, location: str) -> Result:
        """
        Reads a result from the local file system and returns the corresponding `Result` instance.
        Args:
            - location (str): the location to read from
        Returns:
            - Result: a new result instance with the data represented by the location
        """
        new = self.copy()
        new.location = location

        self.logger.debug("Starting to read result from {}...".format(location))

        with open(os.path.join(self.dir, location), "rb") as f:
            value = f.read()

        new.value = self.serializer.deserialize(value)

        self.logger.debug("Finished reading result from {}...".format(location))

        return new


    def write(self, result: typing.Any, input_mapping):
        """Write result to file
        Args:
            result (typing.Any): Write result to file
        """
        if self.input_mapping is not None:
            input_mapping = self.input_mapping
            logger.info("Manual input mapping provided. Overriding")
        if input_mapping is None and self.file_name is not None:
            input_mapping = self.file_name
        elif input_mapping is None and self.file_name is None:
            raise Exception(
                "Both file_name and input_mapping are empty. One of them should be available"
            )
        path_string = Path(self.path, input_mapping).with_suffix(self.file_type)
        loguru.logger.info(f"Wrting result from {path_string}")
        with open(path_string, "wb") as f:
            f.write(self.caching_method.dumps(result, use_bin_type=False))
        loguru.logger.success("Data successfully cached")
