"""Data I/O utilities for loading and saving EMG data."""

import numpy as np
import pandas as pd
import json
import h5py
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
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
    
    @staticmethod
    def load_csv(
        filepath: Union[str, Path],
        sampling_rate: float = 1000.0,
        channel_columns: Optional[List[Union[str, int]]] = None,
        time_column: Optional[Union[str, int]] = None,
        skip_rows: int = 0,
        delimiter: str = ',',
        header: Optional[int] = 0
    ) -> EMGSignal:
        """
        Load multi-channel EMG data from CSV file.
        
        This method supports various CSV formats commonly used for EMG data:
        - Files with column headers
        - Files without headers (specify channel indices)
        - Files with time column (will be used to calculate sampling rate)
        - Multi-channel data in separate columns
        
        Args:
            filepath: Path to CSV file
            sampling_rate: Sampling frequency in Hz (used if time_column not specified)
            channel_columns: List of column names or indices for EMG channels.
                           If None, all columns except time_column are used.
            time_column: Column name or index containing time data.
                        If specified, sampling_rate will be calculated from time data.
            skip_rows: Number of rows to skip at the beginning (e.g., metadata rows)
            delimiter: Column delimiter (default: ',')
            header: Row number to use as column names. None if no header.
            
        Returns:
            EMGSignal object with multi-channel data
            
        Examples:
            >>> # Load CSV with headers, auto-detect channels
            >>> signal = DataLoader.load_csv('data.csv', sampling_rate=1000.0)
            
            >>> # Load CSV with specific channels by name
            >>> signal = DataLoader.load_csv(
            ...     'data.csv',
            ...     channel_columns=['CH1', 'CH2', 'CH3'],
            ...     time_column='Time'
            ... )
            
            >>> # Load CSV without headers, specify channel indices
            >>> signal = DataLoader.load_csv(
            ...     'data.csv',
            ...     channel_columns=[0, 1, 2],
            ...     header=None
            ... )
        """
        filepath = Path(filepath)
        
        # Load CSV file
        df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            header=header,
            skiprows=skip_rows
        )
        
        # Handle time column and calculate sampling rate if provided
        calculated_sr = sampling_rate
        if time_column is not None:
            if time_column in df.columns or isinstance(time_column, int):
                time_data = df[time_column].values
                # Calculate sampling rate from time data
                if len(time_data) > 1:
                    time_diff = np.diff(time_data)
                    avg_time_diff = np.mean(time_diff)
                    calculated_sr = 1.0 / avg_time_diff
                # Remove time column from dataframe
                df = df.drop(columns=[time_column])
        
        # Select channels
        if channel_columns is not None:
            # Use specified channels
            data = df[channel_columns].values
            if header is not None:
                channels = [str(col) for col in channel_columns]
            else:
                channels = [f"Channel_{i}" for i in range(len(channel_columns))]
        else:
            # Use all columns
            data = df.values
            if header is not None:
                channels = [str(col) for col in df.columns]
            else:
                channels = [f"Channel_{i}" for i in range(data.shape[1])]
        
        # Create metadata with file information
        metadata = {
            'source_file': str(filepath),
            'file_format': 'csv',
            'delimiter': delimiter,
            'n_samples': data.shape[0],
            'n_channels': data.shape[1]
        }
        
        return EMGSignal(
            data=data,
            sampling_rate=calculated_sr,
            channels=channels,
            metadata=metadata
        )


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
    
    @staticmethod
    def save_csv(
        emg_signal: EMGSignal,
        filepath: Union[str, Path],
        include_time: bool = True,
        delimiter: str = ','
    ):
        """
        Save EMG data to CSV file.
        
        Args:
            emg_signal: EMG signal to save
            filepath: Output file path
            include_time: Whether to include time column
            delimiter: Column delimiter (default: ',')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(
            emg_signal.data,
            columns=emg_signal.channels
        )
        
        # Add time column if requested
        if include_time:
            time = emg_signal.time_vector
            df.insert(0, 'Time', time)
        
        # Save to CSV
        df.to_csv(filepath, sep=delimiter, index=False)
