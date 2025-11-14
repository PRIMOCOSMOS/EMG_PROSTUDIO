"""Machine learning module for EMG analysis."""

from emg_prostudio.ml.embeddings import (
    EMGEmbeddingModel,
    HuggingFaceEMGModel,
    SimpleLinearProbe,
    create_embedding_model
)

from emg_prostudio.ml.trainers import (
    QualityClassifier,
    FatigueMonitor,
    ModelTrainer
)

__all__ = [
    'EMGEmbeddingModel',
    'HuggingFaceEMGModel',
    'SimpleLinearProbe',
    'create_embedding_model',
    'QualityClassifier',
    'FatigueMonitor',
    'ModelTrainer'
]
