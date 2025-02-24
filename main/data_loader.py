import json
import os
from pathlib import Path
import pyarrow.parquet as pq
from typing import Dict, Generator


class DataLoader:
    """
    A class to handle loading of data and models.

    This class provides methods to stream data from a Parquet file in batches,
    load a pre-trained model from a pickle file, and load a pipeline configuration
    from a JSONC file.
    """

    @staticmethod
    def stream_data(filepath: str, batch_size: int) -> Generator:
        """
        Reads a Parquet file in batches and yields data as pandas DataFrames.

        Args:
            filepath (str): Path to the Parquet file.
            batch_size (int): Number of rows per batch to avoid memory overload.

        Yields:
            pd.DataFrame: A batch of data as a DataFrame.
        """

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
        """
        Loads a pre-trained model from a pickle file.

        Args:
            filepath (str): Path to the model file.

        Returns:
            object: A scikit-learn model loaded from the pickle file.
        """
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

    @staticmethod
    def load_pipeline_file(filepath: str) -> Dict:
        """
        Loads a pipeline configuration from a JSONC file.

        Args:
            filepath (str): Path to the pipeline JSONC file.

        Returns:
            Dict: A dictionary containing the pipeline configuration.
        """
        try:
            with open(filepath, "r") as file:
                lines = [
                    line
                    for line in file.readlines()
                    if not line.strip().startswith("//")
                ]
                pipeline_spec = json.loads("\n".join(lines))
                steps_config = pipeline_spec.get("steps", {})
            return steps_config

        except Exception as e:
            raise RuntimeError(f"Error building pipeline from {filepath}: {e}")
