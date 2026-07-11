from .drifting_model import DriftModel
from .flow_matching import FlowMatching
from .improved_mean_flow import ImprovedMeanFlow
from .mean_flow import MeanFlow

MODEL_REGISTRY = {
    "flow_matching": FlowMatching,
    "mean_flow": MeanFlow,
    "improved_mean_flow": ImprovedMeanFlow,
    "drifting_model": DriftModel,
}

ONE_STEP_METHODS = {"mean_flow", "improved_mean_flow", "drifting_model"}

def get_model(name: str, data_info, device="cpu", **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](data_info, device=device, **kwargs)