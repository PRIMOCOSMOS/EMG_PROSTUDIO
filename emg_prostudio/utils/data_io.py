"""Data I/O utilities for loading and saving EMG data."""

import numpy as np
import json
import h5py
from pathlib import Path
from typing import Union, Dict, Any
from emg_prostudio.core.signal import EMGSignal


class DataLoader:
    """Load EMG data from various formats."""
    
    @staticmethod
    def load_numpy(filepath: Union[str, Path]) -> EMGSignal:
        """
        Load EMG data from numpy file (.npy or .npz).
        
        Args:
            filepath: Path to numpy file
            
        Returns:
            EMGSignal object
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npy':
            data = np.load(filepath)
            # Assume default parameters if not provided
            return EMGSignal(data=data, sampling_rate=1000.0)
        
        elif filepath.suffix == '.npz':
            npz_data = np.load(filepath)
            data = npz_data['data']
            sampling_rate = npz_data.get('sampling_rate', 1000.0)
            channels = npz_data.get('channels', None)
            if channels is not None:
                channels = channels.tolist()
            return EMGSignal(
                data=data,
                sampling_rate=float(sampling_rate),
                channels=channels
            )
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def load_hdf5(filepath: Union[str, Path]) -> EMGSignal:
        """
        Load EMG data from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            EMGSignal object
        """
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            data = f['data'][:]
            sampling_rate = f.attrs.get('sampling_rate', 1000.0)
            
            # Load channels if available
            channels = None
            if 'channels' in f:
                channels = [ch.decode() if isinstance(ch, bytes) else ch 
                           for ch in f['channels'][:]]
            
            # Load metadata if available
            metadata = {}
            if 'metadata' in f.attrs:
                metadata = json.loads(f.attrs['metadata'])
            
            return EMGSignal(
                data=data,
                sampling_rate=float(sampling_rate),
                channels=channels,
                metadata=metadata
            )
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> EMGSignal:
        """
        Load EMG data from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            EMGSignal object
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
        
        return EMGSignal.from_dict(data_dict)


class DataSaver:
    """Save EMG data to various formats."""
    
    @staticmethod
    def save_numpy(emg_signal: EMGSignal, filepath: Union[str, Path]):
        """
        Save EMG data to numpy file (.npz).
        
        Args:
            emg_signal: EMG signal to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            filepath,
            data=emg_signal.data,
            sampling_rate=emg_signal.sampling_rate,
            channels=np.array(emg_signal.channels)
        )
    
    @staticmethod
    def save_hdf5(emg_signal: EMGSignal, filepath: Union[str, Path]):
        """
        Save EMG data to HDF5 file.
        
        Args:
            emg_signal: EMG signal to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save data
            f.create_dataset('data', data=emg_signal.data)
            
            # Save attributes
            f.attrs['sampling_rate'] = emg_signal.sampling_rate
            
            # Save channels
            f.create_dataset('channels', data=np.array(emg_signal.channels, dtype='S'))
            
            # Save metadata
            if emg_signal.metadata:
                f.attrs['metadata'] = json.dumps(emg_signal.metadata)
    
    @staticmethod
    def save_json(emg_signal: EMGSignal, filepath: Union[str, Path]):
        """
        Save EMG data to JSON file.
        
        Args:
            emg_signal: EMG signal to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(emg_signal.to_dict(), f, indent=2)
