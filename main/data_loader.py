import pyarrow.parquet as pq
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

    def load_model(self):
        pass

    def load_pipeline_file(self):
        pass
