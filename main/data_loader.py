import os
import pyarrow.parquet as pq

from pathlib import Path
from typing import Generator


class DataLoader:

    @staticmethod
    def stream_data(filepath: str, batch_size: int) -> Generator:
        try:
            parquet_file = pq.ParquetFile(filepath)

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()
                df.dropna(inplace=True)
                yield df

        except Exception as e:
            raise RuntimeError(f"Error reading Parquet file from {filepath}: {e}")

    @staticmethod
    def load_model(filepath: str) -> object:
        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_script_dir)
            model_path = Path(filepath)
            full_model_path = base_dir / model_path

            with open(full_model_path, "rb") as model_file:
                import pickle
                model = pickle.load(model_file)

            return model

        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {e}")

    def load_pipeline_file(self):
        pass
