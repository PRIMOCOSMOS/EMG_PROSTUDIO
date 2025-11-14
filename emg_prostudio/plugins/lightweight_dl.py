"""Lightweight deep learning plugin.

Implements simple neural network models for EMG analysis:
- Perceptron for binary classification
- Small feedforward networks for feature learning
- Minimal computational requirements
"""

from typing import Dict, Any, Optional
import numpy as np
from emg_prostudio.core.signal import EMGSignal
from emg_prostudio.plugins import AnalysisPlugin

# Try to import PyTorch, but make it optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SimpleMLP(nn.Module if TORCH_AVAILABLE else object):
    """Simple multi-layer perceptron for EMG classification."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 3):
        """
        Initialize MLP.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer size
            output_size: Output dimension (e.g., 3 for full/half/invalid)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class LightweightDeepLearning(AnalysisPlugin):
    """
    Lightweight deep learning plugin.
    
    Uses small neural networks for pattern recognition in EMG signals.
    """
    
    def __init__(self):
        """Initialize lightweight DL plugin."""
        super().__init__("LightweightDeepLearning", "1.0.0")
        self.model = None
        self.device = 'cpu'
        
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Install with: pip install torch")
    
    def analyze(
        self,
        emg_signal: EMGSignal,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform lightweight deep learning analysis.
        
        Args:
            emg_signal: Input EMG signal
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        if not TORCH_AVAILABLE:
            return {
                'method': 'lightweight_dl',
                'error': 'PyTorch not available',
                'message': 'Install PyTorch to use deep learning features'
            }
        
        # Extract features for model input
        from emg_prostudio.features.extractor import FeatureExtractor
        extractor = FeatureExtractor(emg_signal.sampling_rate)
        features = extractor.extract_windowed(
            emg_signal,
            window_size=0.25,
            overlap=0.5
        )
        
        # For demonstration: create simple classification
        # In practice, this would use a trained model
        results = {
            'method': 'lightweight_dl',
            'features': features,
            'message': 'Lightweight DL analysis (demo mode - training required)'
        }
        
        return results
    
    def train_classifier(
        self,
        training_data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Train the classifier on labeled data.
        
        Args:
            training_data: Training features (N x features)
            labels: Training labels (N,)
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
        
        input_size = training_data.shape[1]
        n_classes = len(np.unique(labels))
        
        self.model = SimpleMLP(input_size, hidden_size=32, output_size=n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X = torch.FloatTensor(training_data)
        y = torch.LongTensor(labels)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.name,
            'version': self.version,
            'description': 'Lightweight deep learning models for EMG analysis',
            'type': 'deep_learning',
            'pytorch_available': TORCH_AVAILABLE,
            'capabilities': [
                'feature_learning',
                'classification',
                'pattern_recognition'
            ]
        }
