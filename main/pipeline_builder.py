import importlib

from decouple import config, UndefinedValueError


class PipelineBuilder:

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

    def build(self):
        pass
