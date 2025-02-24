import pandas as pd

from typing import List, Generator, Dict
from data_loader import DataLoader
from logger import Logger


class Scorer:
    def __init__(self, data_path, pipeline_path):
        self.data_path = data_path
        self.data_loader = DataLoader()
        self.logger = Logger()

    @staticmethod
    def _extract_features(record: Dict, features: List) -> List:
        try:
            row = []
            for feat in features:
                if feat not in record:
                    raise RuntimeError(
                        f"Feature '{feat}' is missing from a data record."
                    )
                row.append(record[feat])
            return row

        except Exception as e:
            raise RuntimeError(f"Error while extracting features: {e}")

    def _batch_generator(self, batch_size: int, features: List) -> Generator:
        try:
            batch = []
            self.logger.log_info("Starting batch data yielding...")
            for record in self.data_loader.stream_data(self.data_path, batch_size):
                batch.append(record[features])

                if len(batch) >= batch_size:
                    yield pd.concat(batch[:batch_size], ignore_index=True)
                    batch = batch[batch_size:]

            if batch:
                yield pd.concat(batch, ignore_index=True)

        except Exception as e:
            raise RuntimeError(f"Error while generating batches: {e}")

    def score(self):
        pass
