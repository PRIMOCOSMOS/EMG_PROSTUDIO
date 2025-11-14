"""Plugin system for extensible analysis methods.

Supports three technical paths:
1. Traditional signal processing
2. Lightweight deep learning (perceptron, small neural networks)
3. External model integration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from emg_prostudio.core.signal import EMGSignal


class AnalysisPlugin(ABC):
    """Base class for analysis plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
        """
        self.name = name
        self.version = version
        self.config = {}
    
    @abstractmethod
    def analyze(
        self,
        emg_signal: EMGSignal,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform analysis on EMG signal.
        
        Args:
            emg_signal: Input EMG signal
            **kwargs: Additional parameters
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        pass
    
    def configure(self, config: Dict[str, Any]):
        """Configure plugin parameters."""
        self.config.update(config)


class PluginManager:
    """Manage analysis plugins."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.plugins = {}
    
    def register(self, plugin: AnalysisPlugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        self.plugins[plugin.name] = plugin
        print(f"Registered plugin: {plugin.name} v{plugin.version}")
    
    def unregister(self, name: str):
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name to unregister
        """
        if name in self.plugins:
            del self.plugins[name]
            print(f"Unregistered plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[AnalysisPlugin]:
        """
        Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self.plugins.get(name)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered plugins.
        
        Returns:
            Dictionary of plugin information
        """
        return {
            name: plugin.get_info()
            for name, plugin in self.plugins.items()
        }
    
    def analyze(
        self,
        plugin_name: str,
        emg_signal: EMGSignal,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run analysis using specific plugin.
        
        Args:
            plugin_name: Name of plugin to use
            emg_signal: Input EMG signal
            **kwargs: Additional parameters for plugin
            
        Returns:
            Analysis results
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is None:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        return plugin.analyze(emg_signal, **kwargs)
