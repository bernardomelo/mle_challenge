# README

## Shape's Hard Skill Test - Machine Learning Engineer

This project is a refactored and modularized version of the `job_test_challenge.py` script, designed to meet the requirements of Shape's Machine Learning Engineer challenge. The goal was to refactor the code into a more maintainable, scalable, and production-ready format, while adhering to best practices and ensuring it can handle large-scale data processing scenarios.

---

## Project Structure

1. The project is structured as follows:
    ```plaintext
    shape-mle-challenge/
    ├── artifacts/
    │   ├── model.pkl
    │   └── pipeline.jsonc
    ├── data/
    │   └── dataset.parquet
    ├── main/
    │   ├── logs/
    │   ├── .env
    │   ├── data_loader.py
    │   ├── job_test_challenge.py
    │   ├── logger.py
    │   ├── pipeline_builder.py
    │   ├── run_score_pipeline.py
    │   └── scorer.py
    ├── output/
    ├── .gitignore
    ├── requirements.txt


---

## Key Changes and Refactoring

### 1. **Modularization**
   - The original script was refactored into multiple modules to improve readability, maintainability, and reusability:
     - **`data_loader.py`**: Handles data and model loading.
     - **`pipeline_builder.py`**: Dynamically builds a scikit-learn pipeline based on a configuration file.
     - **`scorer.py`**: Manages the scoring process, including batch processing and parallel execution.
     - **`logger.py`**: Provides a logging utility with colorized console output and file logging.
     - **`run_score_pipeline.py`**: The main script that orchestrates the scoring process.

### 2. **Configuration Management**
   - Configuration parameters such as file paths, batch size, and number of workers are managed via a `.env` file. This allows for easy configuration changes without modifying the code.
   - The pipeline configuration is loaded from a JSONC file (`pipeline.jsonc`), making it easy to modify the pipeline steps without code changes.

### 3. **Dynamic Pipeline Building**
   - The PipelineBuilder class dynamically instantiates sklearn pipeline steps from a JSONC configuration file. This allows for flexible pipeline configuration without modifying the code.

### 4. **Batch Processing**
   - The `DataLoader` class streams data from a Parquet file in batches to avoid memory overload, which is crucial for handling large datasets.
   - The `Scorer` class processes data in parallel using a `ThreadPoolExecutor`, allowing for efficient scoring of large datasets.

### 5. **Logging**
   - A custom `Logger` class was implemented to provide colorized console output and file logging. This improves debugging and monitoring in a production environment.

### 6. **Error Handling**
   - Robust error handling was added throughout the code to ensure that the pipeline can gracefully handle and log errors, making it more suitable for production use.

### 7. **Documentation**
   - Each class and method is documented with clear docstrings, explaining their purpose, arguments, and return values.
   - A `README.md` file was created to provide an overview of the project, the changes made, and the reasoning behind them.

---

## Running the Project

### Prerequisites
- Python 3.13
- scikit-learn 1.6.1
- pandas
- numpy
- pyarrow
- Required libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/shape-mle-challenge.git

2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt

3. Navigate to the project directory:
    ```bash
    cd shape-mle-challenge/main

### Execution
1. Ensure that the .env file is properly configured with the necessary paths and parameters, if not running from cli.
- DATA_PATH='../data/file.parquet'
- PIPELINE_PATH='../artifacts/file.jsonc'
- BATCH_SIZE=int
- MAX_WORKERS=int
- SKLEARN_MODULES='modules separated with comma: sklearn.preprocessing,sklearn.feature_selection,sklearn.decomposition,sklearn.ensemble'

2. Run the scoring process:
    ```bash
   python run_score_pipeline.py

3. Alternatively, you can provide command-line arguments:
    ```bash
   python run_score_pipeline.py --data_path path/to/data.parquet --pipeline_path path/to/pipeline.jsonc --batch_size 1000 --max_workers 4

---

## Future Improvements
1. Unit Tests: Add unit tests for each module to ensure code reliability and facilitate future development. Was very short o time so I had to speed up the development proccess.

2. Integration with Big Data Tools: Integrate with tools like Apache Spark for distributed processing of very large datasets.

3. Model Versioning: Add support for model versioning to track different versions of the model and pipeline configurations.

4. Dockerization: Containerize the application using Docker to simplify deployment and ensure consistency across environments. My working PC is on an outdated Windows 10 version, Docker wouldn't even install.

