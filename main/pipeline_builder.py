import importlib
from typing import Dict

from sklearn.pipeline import Pipeline
from decouple import config, UndefinedValueError


class PipelineBuilder:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.pipeline_config = None

    @staticmethod
    def _instantiate_step(step_name: str, step_params: dict) -> tuple:
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
                        return step_name, cls(**params)

                    except (ModuleNotFoundError, AttributeError):
                        continue

                raise ValueError(f"Unsupported or missing step: {class_name}")

        except Exception as e:
            raise RuntimeError(f"Error while trying to instantiate pipeline step: {e}")

    def _instantiate_pipeline(self) -> Pipeline:
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
        try:
            self.pipeline_config = configs
            return self._instantiate_pipeline()

        except Exception as e:
            raise RuntimeError(e)
