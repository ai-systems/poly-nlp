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
        read_cache: bool = True,
        file_type: str = ".msgpack",
    ):
        self.read_cache = read_cache
        self.path = Path(path)
        self.folder_path = Path(path, task_name)
        if not self.path.exists():
            logger.info(f"Checkpoint dir {self.path} does not exist. Creating one")
            self.path.mkdir(parents=True)
        self.path_string = Path(path, task_name).with_suffix(file_type)

    def read(self, **kwargs):
        """Read from file
        """
        if not os.path.isfile(self.path_string) or not self.read_cache:
            raise FileNotFoundError
        loguru.logger.info(f"Reading result from {self.path_string}")
        with open(self.path_string, "rb") as f:
            return msgpack.unpackb(f.read(), raw=False)

    def write(self, result: typing.Any):
        """Write result to file
        Args:
            result (typing.Any): Write result to file
        """
        loguru.logger.info(f"Wrting result from {self.path_string}")
        with open(self.path_string, "wb") as f:
            f.write(msgpack.packb(result, use_bin_type=False))
        loguru.logger.success("Data successfully cached")
