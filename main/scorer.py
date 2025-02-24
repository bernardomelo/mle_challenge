import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import List, Generator, Dict, Optional
from data_loader import DataLoader
from logger import Logger
from pipeline_builder import PipelineBuilder, Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


class Scorer:
    def __init__(self, data_path, pipeline_path):
        self.data_path = data_path
        self.pipeline_path = pipeline_path
        self.data_loader = DataLoader()
        self.logger = Logger()
        self.model = None

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

    def _process_batch(self, batch: pd.DataFrame, pipeline: Pipeline) -> Optional:
        try:
            transformed_batch = pipeline.fit_transform(batch)
            if not hasattr(self.model, "predict"):
                raise RuntimeError("Loaded model does not have a predict method.")
            return self.model.predict(transformed_batch)
        except Exception as e:
            self.logger.log_fail(f"Error processing batch: {e}")
            return None

    def score(self, batch_size: int = 1000, max_workers: int = 4) -> str:
        try:
            features = ["vibration_x", "vibration_y", "vibration_z"]

            pipeline_configs = self.data_loader.load_pipeline_file(self.pipeline_path)
            self.logger.log_success("Pipeline file loaded successfully.")

            model_filepath = pipeline_configs.pop("model", None)
            pipeline_builder = PipelineBuilder(self.data_loader)

            self.model = self.data_loader.load_model(model_filepath)
            self.logger.log_success("Model file loaded successfully.")

            pipeline = pipeline_builder.build(pipeline_configs)
            self.logger.log_success("Successfully instantiated pipeline parameters.")

            self.logger.log_info("Starting data batch processing and scoring...")

            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_script_dir)
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            today_date = datetime.now().strftime("%Y-%m-%d")
            output_file = base_dir / output_dir / f"predictions-{today_date}.parquet"

            if output_file.exists():
                self.logger.log_warning(
                    f"Output file {output_file} already exists. It will be overwritten."
                )

            with pq.ParquetWriter(
                    output_file, pa.schema([("prediction", pa.int64())])
            ) as writer:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for batch in self._batch_generator(batch_size, features):
                        futures.append(
                            executor.submit(self._process_batch, batch, pipeline)
                        )

                    for future in as_completed(futures):
                        try:
                            predictions = future.result()
                            if predictions is not None:
                                table = pa.Table.from_arrays(
                                    [pa.array(predictions)], names=["prediction"]
                                )
                                writer.write_table(table)

                        except Exception as e:
                            self.logger.log_fail(f"Error in batch processing: {e}")

            self.logger.log_success(
                f"Successfully scored data and saved predictions to {output_file}."
            )
            return str(output_file)

        except Exception as e:
            self.logger.log_fail(f"Scoring process failed: {e}")
            raise

