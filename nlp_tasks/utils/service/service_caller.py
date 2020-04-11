import json
import math
from functools import reduce
from typing import Dict, List

import ray
import requests
from loguru import logger
from overrides import overrides
from tqdm import tqdm
from tqdm.contrib import tzip

from covid_fact_bank.utils.data_manipulation.data_manipulation import create_dict_chunks


class ServiceCaller:
    @ray.remote
    def service_post(pos, api_url, request_input):
        extracted = {}
        for id, input_data in tqdm(
            request_input.items(), desc="Requesting server", position=pos
        ):
            try:
                data = requests.post(
                    api_url,
                    json=input_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                extracted[id] = data.json()
            except json.decoder.JSONDecodeError:
                extracted[id] = data.json()
                logger.warning(f"Unable to parse {input_data}")

        return extracted

    @staticmethod
    def to_iterator(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    def run(self, parameters: List[Dict], end_point: str, input_parsed: Dict) -> List:
        """Run the service being requested using the end point and with request_input.
        
        Arguments:
            Task {[type]} -- [description]
            parameters {List[Dict]} -- List of Dict[{port, host and number of workers}]
            end_point {String} -- string to end_point
            input_parsed {Dict} -- input that will be sent as request
        
        Returns:
            List -- list of results from the service
        """
        ray.init()

        api_urls = [
            f"http://{parameters['host']}:{p}/{end_point}" for p in parameters["port"]
        ]

        logger.info(f"End_point = {end_point}")

        batch_size = math.ceil(len(input_parsed) / len(api_urls))

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {len(api_urls)}")

        batches = create_dict_chunks(input_parsed, batch_size)

        logger.info("Running service.")

        batch_results = ray.get(
            [
                self.service_post.remote(pos, url, batch)
                for pos, (url, batch) in enumerate(zip(api_urls, batches))
            ]
        )

        return reduce(lambda x, y: {**x, **y}, batch_results, {})
