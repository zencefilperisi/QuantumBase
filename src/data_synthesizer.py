# src/data_synthesizer.py

import numpy as np
import pandas as pd
from src.physics_engine import TunnelingPhysics
from src.noise_generator import NoiseGenerator

def generate_synthetic_dataset(n_samples=5000):
    """
    Accelerated dataset generation using NumPy vectorization.
    Optimized for high-precision base calling (A, T, C, G).
    """
    physics = TunnelingPhysics()
    noise_gen = NoiseGenerator()
    bases = ['A', 'T', 'C', 'G']
    
    # 1. Randomly sample bases
    selected_bases = np.random.choice(bases, n_samples)
    
    # 2. Map bases to their physical tunneling current values
    current_map = {base: physics.get_base_current(base) for base in bases}
    clean_currents = np.array([current_map[b] for b in selected_bases])
    
    # 3. Create signal matrix (n_samples x window_size)
    # Using a window size of 120 for better noise averaging and C-T resolution
    window_size = 120
    signals = np.tile(clean_currents[:, np.newaxis], (1, window_size))
    
    # 4. Inject composite noise in a vectorized manner
    noisy_signals = noise_gen.add_noise(signals)
    
    # 5. Extract features using matrix operations (High Speed)
    means = np.mean(noisy_signals, axis=1)
    stds = np.std(noisy_signals, axis=1)
    diffs = np.mean(np.abs(np.diff(noisy_signals, axis=1)), axis=1)
    ptps = np.ptp(noisy_signals, axis=1)
    
    # Spectral analysis (FFT) for C-T discrimination
    fft_vals = np.abs(np.fft.rfft(noisy_signals, axis=1))
    peak_freqs = np.argmax(fft_vals[:, 1:], axis=1)
    
    # 6. Build the final dataframe
    df = pd.DataFrame({
        'mean': means,
        'std': stds,
        'max': np.max(noisy_signals, axis=1),
        'min': np.min(noisy_signals, axis=1),
        'snr': means / np.where(stds == 0, 1e-12, stds),
        'mean_diff': diffs,
        'peak_freq': peak_freqs,
        'label': selected_bases
    })
    
    print(f"Status: Successfully synthesized {n_samples} samples with 7D features.")
    return df

if __name__ == "__main__":
    df = generate_synthetic_dataset(n_samples=4000)
    print(df.head())