import importlib
from typing import Dict
from logger import Logger

from sklearn.pipeline import Pipeline
from decouple import config, UndefinedValueError


class PipelineBuilder:
    """
    A class to build a scikit-learn pipeline from a configuration file.

    This class dynamically instantiates pipeline steps based on a JSONC configuration
    file and constructs a scikit-learn Pipeline object.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.pipeline_config = None
        self.logger = Logger()

    def _instantiate_step(self, step_name: str, step_params: Dict) -> tuple:
        """
        Dynamically imports and instantiates a scikit-learn transformer.

        Args:
            step_name (str): The name of the pipeline step.
            step_params (Dict): Parameters for the step.

        Returns:
            tuple: A tuple containing the step name and the instantiated object.

        Note:
            Parameters matching their default values (e.g., `degree=2` for `PolynomialFeatures`)
            are not displayed in the logs by default. To address this, we explicitly log the parameters.
        """
        try:
            sklearn_modules = config(
                "SKLEARN_MODULES",
                default="sklearn.preprocessing,sklearn.feature_selection,sklearn.decomposition,sklearn.ensemble",
            ).split(",")

        except UndefinedValueError:
            raise ValueError("No sklearn modules for pipeline found on .env file.")

        try:
            for class_name, params in step_params.items():
                for module_name in sklearn_modules:
                    try:
                        module = importlib.import_module(module_name)
                        cls = getattr(module, class_name)
                        self.logger.log_info(
                            f"Instantiating {class_name} with params: {params}"
                        )
                        return step_name, cls(**params)
                    except (ModuleNotFoundError, AttributeError):
                        continue

                raise ValueError(f"Unsupported or missing step: {class_name}")

        except Exception as e:
            raise RuntimeError(f"Error while trying to instantiate pipeline step: {e}")

    def _instantiate_pipeline(self) -> Pipeline:
        """
        Dynamically instantiates the pipeline steps based on the loaded configuration.

        Returns:
            Pipeline: A scikit-learn Pipeline object.
        """

        try:
            if not self.pipeline_config:
                raise ValueError(
                    "Pipeline configuration not loaded. Call load_pipeline() first."
                )

            steps_instances = []
            for step_name, step_params in self.pipeline_config.items():
                try:
                    step = self._instantiate_step(step_name, step_params)
                    steps_instances.append(step)
                except ValueError as e:
                    print(f"Skipping unsupported step: {step_name} - {e}")

            return Pipeline(steps_instances)

        except Exception as e:
            raise RuntimeError(f"Error while trying to instantiate pipeline: {e}")

    def build(self, configs: Dict) -> Pipeline:
        """
        Builds and returns the scikit-learn pipeline.

        Args:
            configs (Dict): The pipeline configuration.

        Returns:
            Pipeline: A scikit-learn Pipeline object.
        """
        try:
            self.pipeline_config = configs
            return self._instantiate_pipeline()

        except Exception as e:
            raise RuntimeError(e)
