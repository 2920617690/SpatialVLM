from .schema import AVVSample, RelationSample, SceneObject, SubClaim, TrajectoryStep, load_avv_samples
from .synthetic_generator import generate_synthetic_dataset

try:
    from .multimodal_dataset import AVVQCRDataset, AVVSupervisedDataset, QwenChatCollator
except ModuleNotFoundError:
    AVVQCRDataset = None
    AVVSupervisedDataset = None
    QwenChatCollator = None

__all__ = [
    "AVVQCRDataset",
    "AVVSample",
    "AVVSupervisedDataset",
    "QwenChatCollator",
    "RelationSample",
    "SceneObject",
    "SubClaim",
    "TrajectoryStep",
    "generate_synthetic_dataset",
    "load_avv_samples",
]
