"""
Lightweight machine learning models for EMG analysis.

Implements LightGBM and RandomForest models for:
- Quality classification (full/half/invalid movements)
- Fatigue monitoring and prediction

These models work with hybrid features (traditional + deep learning embeddings).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path
import pickle


class QualityClassifier:
    """
    LightGBM-based quality classifier for movement quality assessment.
    
    Classifies each rep into:
    - Full range (>80% amplitude)
    - Half range (40-80% amplitude)
    - Invalid (<40% amplitude)
    """
    
    def __init__(self, use_lgbm: bool = True):
        """
        Initialize quality classifier.
        
        Args:
            use_lgbm: If True, use LightGBM; else use RandomForest
        """
        self.use_lgbm = use_lgbm
        self.model = None
        self.is_trained = False
        self.feature_names = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Train the quality classifier.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (0=invalid, 1=half, 2=full)
            feature_names: Names of features
            **kwargs: Additional parameters for the classifier
        """
        self.feature_names = feature_names
        
        if self.use_lgbm:
            try:
                import lightgbm as lgb
                
                # Default parameters optimized for small datasets
                params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
                params.update(kwargs)
                
                # Create dataset
                train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
                
                # Train
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10)]
                )
                
                self.is_trained = True
                print("✓ LightGBM quality classifier trained")
                
            except ImportError:
                warnings.warn("LightGBM not installed. Falling back to RandomForest")
                self._train_rf(X_train, y_train, **kwargs)
        else:
            self._train_rf(X_train, y_train, **kwargs)
    
    def _train_rf(self, X_train, y_train, **kwargs):
        """Train RandomForest as fallback."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            params.update(kwargs)
            
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)
            
            self.is_trained = True
            print("✓ RandomForest quality classifier trained")
            
        except ImportError:
            warnings.warn("scikit-learn not installed. Cannot train classifier")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quality labels.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted labels (0=invalid, 1=half, 2=full)
        """
        if not self.is_trained or self.model is None:
            warnings.warn("Model not trained. Returning default predictions")
            return np.zeros(len(X), dtype=int)
        
        if self.use_lgbm and hasattr(self.model, 'predict'):
            # LightGBM
            predictions = self.model.predict(X)
            return np.argmax(predictions, axis=1)
        else:
            # RandomForest
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities (n_samples, 3)
        """
        if not self.is_trained or self.model is None:
            warnings.warn("Model not trained. Returning uniform probabilities")
            return np.ones((len(X), 3)) / 3
        
        if self.use_lgbm and hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'use_lgbm': self.use_lgbm,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.use_lgbm = data['use_lgbm']
        self.is_trained = data['is_trained']
        self.feature_names = data.get('feature_names')
        print(f"✓ Model loaded from {filepath}")


class FatigueMonitor:
    """
    RandomForest-based fatigue monitoring model.
    
    Predicts fatigue level (0-1 scale) based on extracted features.
    Can track fatigue progression over time.
    """
    
    def __init__(self, use_rf: bool = True):
        """
        Initialize fatigue monitor.
        
        Args:
            use_rf: If True, use RandomForest; else use LightGBM
        """
        self.use_rf = use_rf
        self.model = None
        self.is_trained = False
        self.feature_names = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Train the fatigue monitoring model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training fatigue scores (0-1, where 1 is most fatigued)
            feature_names: Names of features
            **kwargs: Additional parameters
        """
        self.feature_names = feature_names
        
        if self.use_rf:
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                params = {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
                params.update(kwargs)
                
                self.model = RandomForestRegressor(**params)
                self.model.fit(X_train, y_train)
                
                self.is_trained = True
                print("✓ RandomForest fatigue monitor trained")
                
            except ImportError:
                warnings.warn("scikit-learn not installed. Falling back to LightGBM")
                self._train_lgbm(X_train, y_train, **kwargs)
        else:
            self._train_lgbm(X_train, y_train, **kwargs)
    
    def _train_lgbm(self, X_train, y_train, **kwargs):
        """Train LightGBM as alternative."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            params.update(kwargs)
            
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
            
            self.is_trained = True
            print("✓ LightGBM fatigue monitor trained")
            
        except ImportError:
            warnings.warn("LightGBM not installed. Cannot train fatigue monitor")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fatigue scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted fatigue scores (0-1)
        """
        if not self.is_trained or self.model is None:
            warnings.warn("Model not trained. Returning default predictions")
            return np.zeros(len(X))
        
        if self.use_rf:
            predictions = self.model.predict(X)
        else:
            predictions = self.model.predict(X)
        
        # Clip to [0, 1] range
        return np.clip(predictions, 0, 1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}
        
        if self.use_rf and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'feature_importance'):
            importances = self.model.feature_importance()
        else:
            return {}
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        return dict(zip(feature_names, importances))
    
    def save(self, filepath: str):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'use_rf': self.use_rf,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.use_rf = data['use_rf']
        self.is_trained = data['is_trained']
        self.feature_names = data.get('feature_names')
        print(f"✓ Model loaded from {filepath}")


class ModelTrainer:
    """
    Unified trainer for both quality and fatigue models.
    
    Handles the complete training pipeline:
    1. Feature extraction
    2. Data splitting
    3. Model training
    4. Evaluation
    5. Model saving
    """
    
    def __init__(self):
        """Initialize model trainer."""
        self.quality_model = None
        self.fatigue_model = None
    
    def train_quality_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        use_lgbm: bool = True
    ) -> QualityClassifier:
        """
        Train quality classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names
            use_lgbm: Use LightGBM (True) or RandomForest (False)
            
        Returns:
            Trained quality classifier
        """
        model = QualityClassifier(use_lgbm=use_lgbm)
        model.train(X_train, y_train, feature_names=feature_names)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == y_val)
            print(f"✓ Validation accuracy: {accuracy:.3f}")
        
        self.quality_model = model
        return model
    
    def train_fatigue_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        use_rf: bool = True
    ) -> FatigueMonitor:
        """
        Train fatigue monitoring model.
        
        Args:
            X_train: Training features
            y_train: Training fatigue scores
            X_val: Validation features (optional)
            y_val: Validation scores (optional)
            feature_names: Feature names
            use_rf: Use RandomForest (True) or LightGBM (False)
            
        Returns:
            Trained fatigue monitor
        """
        model = FatigueMonitor(use_rf=use_rf)
        model.train(X_train, y_train, feature_names=feature_names)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            predictions = model.predict(X_val)
            mse = np.mean((predictions - y_val) ** 2)
            rmse = np.sqrt(mse)
            print(f"✓ Validation RMSE: {rmse:.4f}")
        
        self.fatigue_model = model
        return model
