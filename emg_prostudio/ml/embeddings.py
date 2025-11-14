"""
Deep learning embedding extractor for EMG signals.

Supports loading pre-trained models from various sources including HuggingFace.
Uses linear probe technique to extract embeddings that can be combined with
traditional features.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
import warnings


class EMGEmbeddingModel:
    """
    Base class for EMG embedding models.
    
    This provides a standard interface for extracting embeddings from
    pre-trained models, regardless of the underlying architecture.
    """
    
    def __init__(self, model_name: str, embedding_dim: int = 128):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name/identifier of the model
            embedding_dim: Dimension of the embedding vectors
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = None
    
    def extract_embeddings(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from EMG signal.
        
        Args:
            signal_data: EMG signal data (n_samples, n_channels) or (n_samples,)
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        raise NotImplementedError("Subclasses must implement extract_embeddings")
    
    def load_model(self):
        """Load the pre-trained model."""
        raise NotImplementedError("Subclasses must implement load_model")


class HuggingFaceEMGModel(EMGEmbeddingModel):
    """
    Wrapper for HuggingFace pre-trained EMG models.
    
    Supports loading models from HuggingFace with mirror site option for
    regions with restricted access.
    """
    
    def __init__(
        self,
        model_id: str,
        embedding_dim: int = 128,
        use_mirror: bool = True,
        mirror_url: str = "https://hf-mirror.com"
    ):
        """
        Initialize HuggingFace EMG model.
        
        Args:
            model_id: HuggingFace model identifier
            embedding_dim: Dimension of embeddings
            use_mirror: Whether to use mirror site
            mirror_url: URL of HuggingFace mirror (default: hf-mirror.com)
        """
        super().__init__(model_id, embedding_dim)
        self.model_id = model_id
        self.use_mirror = use_mirror
        self.mirror_url = mirror_url
        
    def load_model(self):
        """
        Load model from HuggingFace (or mirror).
        
        This is a placeholder implementation. Actual implementation would
        depend on the specific model architecture being used.
        """
        try:
            # Set environment variable for mirror if needed
            if self.use_mirror:
                import os
                os.environ['HF_ENDPOINT'] = self.mirror_url
            
            # Try to import transformers
            try:
                from transformers import AutoModel, AutoConfig
                
                print(f"Loading model from HuggingFace: {self.model_id}")
                if self.use_mirror:
                    print(f"Using mirror: {self.mirror_url}")
                
                # Load model
                # Note: Actual model loading depends on the specific EMG model
                # This is a placeholder structure
                config = AutoConfig.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
                self.model.eval()  # Set to evaluation mode
                
                print(f"✓ Model loaded successfully")
                return True
                
            except ImportError:
                warnings.warn(
                    "transformers library not installed. "
                    "Install with: pip install transformers"
                )
                return False
                
        except Exception as e:
            warnings.warn(f"Failed to load HuggingFace model: {e}")
            return False
    
    def extract_embeddings(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract embeddings using the HuggingFace model.
        
        Args:
            signal_data: EMG signal data
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            # Return zero embeddings if model not loaded
            warnings.warn("Model not loaded, returning zero embeddings")
            return np.zeros(self.embedding_dim)
        
        try:
            import torch
            
            # Convert to tensor
            if signal_data.ndim == 1:
                signal_data = signal_data.reshape(1, -1)
            
            # Normalize signal
            signal_tensor = torch.from_numpy(signal_data).float()
            signal_tensor = (signal_tensor - signal_tensor.mean()) / (signal_tensor.std() + 1e-8)
            
            # Extract embeddings (architecture-specific)
            with torch.no_grad():
                # This is a placeholder - actual implementation depends on model
                # outputs = self.model(signal_tensor)
                # embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = np.random.randn(self.embedding_dim)  # Placeholder
            
            return embeddings
            
        except Exception as e:
            warnings.warn(f"Failed to extract embeddings: {e}")
            return np.zeros(self.embedding_dim)


class SimpleLinearProbe(EMGEmbeddingModel):
    """
    Simple linear probe model for extracting embeddings.
    
    This is a lightweight alternative that can be trained on small datasets.
    Uses a simple neural network to project EMG signals into an embedding space.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: list = None
    ):
        """
        Initialize linear probe.
        
        Args:
            input_dim: Input dimension (signal length or feature dimension)
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__("linear_probe", embedding_dim)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]
        
    def load_model(self):
        """Create and initialize the linear probe network."""
        try:
            import torch
            import torch.nn as nn
            
            class LinearProbe(nn.Module):
                def __init__(self, input_dim, embedding_dim, hidden_dims):
                    super().__init__()
                    layers = []
                    
                    # Input layer
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.2))
                        prev_dim = hidden_dim
                    
                    # Output layer
                    layers.append(nn.Linear(prev_dim, embedding_dim))
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            self.model = LinearProbe(self.input_dim, self.embedding_dim, self.hidden_dims)
            print(f"✓ Linear probe initialized")
            return True
            
        except ImportError:
            warnings.warn("PyTorch not installed. Install with: pip install torch")
            return False
    
    def extract_embeddings(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract embeddings using linear probe.
        
        Args:
            signal_data: EMG signal data
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            return np.zeros(self.embedding_dim)
        
        try:
            import torch
            
            # Flatten and normalize input
            signal_flat = signal_data.flatten()
            
            # Resize to match input dimension
            if len(signal_flat) > self.input_dim:
                signal_flat = signal_flat[:self.input_dim]
            elif len(signal_flat) < self.input_dim:
                signal_flat = np.pad(signal_flat, (0, self.input_dim - len(signal_flat)))
            
            # Convert to tensor
            signal_tensor = torch.from_numpy(signal_flat).float()
            signal_tensor = (signal_tensor - signal_tensor.mean()) / (signal_tensor.std() + 1e-8)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model(signal_tensor.unsqueeze(0))
                embeddings = embeddings.squeeze(0).numpy()
            
            return embeddings
            
        except Exception as e:
            warnings.warn(f"Failed to extract embeddings: {e}")
            return np.zeros(self.embedding_dim)


def create_embedding_model(
    model_type: str = "simple",
    **kwargs
) -> Optional[EMGEmbeddingModel]:
    """
    Factory function to create embedding models.
    
    Args:
        model_type: Type of model ('huggingface', 'simple', or 'none')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        EMGEmbeddingModel instance or None
    """
    if model_type == "huggingface":
        model_id = kwargs.get('model_id', 'default-emg-model')
        return HuggingFaceEMGModel(model_id=model_id, **kwargs)
    
    elif model_type == "simple":
        input_dim = kwargs.get('input_dim', 1000)
        embedding_dim = kwargs.get('embedding_dim', 128)
        return SimpleLinearProbe(input_dim=input_dim, embedding_dim=embedding_dim)
    
    elif model_type == "none":
        return None
    
    else:
        warnings.warn(f"Unknown model type: {model_type}, using None")
        return None
