import os
import typing
from pathlib import Path

import loguru
import msgpack
from loguru import logger
from prefect.engine.result_handlers import result_handler


class MsgPackResultHandler(result_handler.ResultHandler):
    def __init__(
        self,
        path: str,
        task_name: str,
        input_mapping=None,
        file_name=None,
        read_cache: bool = True,
        file_type: str = ".msgpack",
    ):
        self.read_cache = read_cache
        self.path = Path(path, task_name)
        self.folder_path = Path(path, task_name)
        self.file_type = file_type
        self.file_name = file_name
        self.input_mapping = input_mapping
        if not self.path.exists():
            logger.info(f"Checkpoint dir {self.path} does not exist. Creating one")
            self.path.mkdir(parents=True)

    def read(self, input_mapping=None, **kwargs):
        """Read from file
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
        loguru.logger.info(f"Reading result from {path_string}")
        if not os.path.isfile(path_string) or not self.read_cache:
            logger.info(f"Cached file not found. Skippig cache retrieval")
            raise FileNotFoundError

        with open(path_string, "rb") as f:
            output = msgpack.unpackb(f.read(), raw=False)
        logger.success("Successfully retrieved from cache")
        return output

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
            f.write(msgpack.packb(result, use_bin_type=False))
        loguru.logger.success("Data successfully cached")
