# src/signal_processor.py

import numpy as np
from scipy.signal import medfilt
try:
    import h5py # Required for reading FAST5 files.
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

class SignalProcessor:
    """
    Advanced Signal Processing Unit for Genomic Deep Learning.
    Implements Z-Score normalization and Sliding Window Variance for event isolation.
    Now includes FAST5 data extraction capabilities for real-world integration.
    """
    def __init__(self, sampling_rate=10000):
        self.sampling_rate = sampling_rate

    def read_fast5(self, file_path):
        """
        [NEW] Reads raw tunneling current from a real Nanopore FAST5 file.
        Requires 'h5py' library.
        """
        if not H5PY_AVAILABLE:
            raise ImportError("FAST5 reading requires 'h5py'. Please run: pip install h5py")
            
        try:
            with h5py.File(file_path, 'r') as f:
                # ONT (Oxford Nanopore) accessing the raw signal in a standard file structure.
                read_group = f['Raw/Reads']
                read_id = list(read_group.keys())[0]
                raw_signal = f[f'Raw/Reads/{read_id}/Signal'][()]
                return raw_signal.astype(np.float32)
        except Exception as e:
            print(f"Error accessing FAST5 content: {e}")
            return None

    def normalize_signal(self, raw_signal):
        """
        Applies Z-Score Normalization. 
        Standardizes signals to Mean=0 and Std=1 to ensure LSTM convergence.
        """
        mean = np.mean(raw_signal)
        std = np.std(raw_signal) + 1e-9
        return (raw_signal - mean) / std

    def apply_median_filter(self, signal, kernel_size=5):
        """
        Removes stochastic 'salt-and-pepper' noise while preserving 
        the sharp transitions (edges) of DNA translocation events.
        """
        return medfilt(signal, kernel_size=kernel_size)


    def detect_events_by_variance(self, signal, window_size=30, threshold=1.2):
        """
        Segmentation using Moving Variance. 
        Highly effective for identifying DNA molecules in a noisy nanogap.
        """
        segments = []
        for i in range(0, len(signal) - window_size, 10):
            window = signal[i : i + window_size]
            if np.var(window) > threshold:
                segments.append(window)
        return segments

    def reshape_for_dl(self, segments, time_steps=30):
        """
        Reshapes extracted segments into a 3D Tensor format:
        [Batch_Size, Time_Steps, Features] required by PyTorch LSTM.
        """
        if not segments or len(segments) == 0:
            return np.array([])
        
        data = np.array(segments)
        # Deep Learning models expect (Samples, Seq_Length, Input_Dim)
        # Ensure data is 2D before reshaping to 3D
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        return data.reshape((data.shape[0], time_steps, 1))