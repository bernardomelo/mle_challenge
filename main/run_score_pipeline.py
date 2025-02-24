import argparse
from scorer import Scorer
from logger import Logger
from decouple import config, UndefinedValueError


def main(
    data_path: str = None,
    pipeline_path: str = None,
    batch_size: int = None,
    max_workers: int = None,
):
    """
    Main function to run the scoring process.

    Args:
        data_path (str): Path to the input data file (Parquet format).
        pipeline_path (str): Path to the pipeline configuration file (JSONC format).
        batch_size (int): Number of records to process in one batch.
        max_workers (int): Maximum number of parallel workers.
    """
    logger = Logger()

    if data_path is None or pipeline_path is None:
        try:
            data_path = config("DATA_PATH")
            pipeline_path = config("PIPELINE_PATH")
            logger.log_info("Paths loaded from .env file.")

            if batch_size is None or max_workers is None:
                batch_size = int(config("BATCH_SIZE"))
                max_workers = int(config("MAX_WORKERS"))
                logger.log_info("Parameters loaded from .env file.")

        except UndefinedValueError:
            logger.log_fail("No paths/parameters provided or found on .env file.")
            raise ValueError("No paths/parameters provided or found on .env file.")

    logger.log_info("Starting scoring process...")
    try:
        scorer = Scorer(data_path, pipeline_path)

        predictions = scorer.score(batch_size=batch_size, max_workers=max_workers)

        if predictions is not None:
            logger.log_success(f"Successfully scored data from {data_path}.")
            print("Predictions:", predictions)
        else:
            logger.log_fail("Scoring failed. No predictions were made.")

    except Exception as e:
        logger.log_fail(f"Scoring process failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scoring process.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the input data file (Parquet format).",
    )
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default=None,
        help="Path to the pipeline configuration file (JSONC format).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of records to process in one batch.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers.",
    )

    args = parser.parse_args()

    main(args.data_path, args.pipeline_path, args.batch_size, args.max_workers)
