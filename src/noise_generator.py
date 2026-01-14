# src/noise_generator.py

import numpy as np

class NoiseGenerator:
    """
    Advanced Noise Generator optimized for matrix operations.
    Handles Thermal, Shot, and Flicker noise for high-throughput simulations.
    """
    def __init__(self, temperature=300, bandwidth=1e6):
        self.k_B = 1.38e-23  # Boltzmann constant
        self.q = 1.602e-19   # Elementary charge
        self.T = temperature
        self.B = bandwidth

    def add_noise(self, current_matrix):
        """
        Infects the current matrix with combined noise sources.
        Supports both 1D arrays and 2D matrices (n_samples, window_size).
        """
        # Ensure input is a numpy array
        current = np.array(current_matrix)
        
        # Calculate resistance (V=1.0V assumed) to determine thermal noise
        # Using 1e-12 as safety floor to avoid division by zero
        resistance = 1.0 / np.where(current == 0, 1e-12, current)
        
        # Apply all noise sources
        noisy_signal = self.apply_all_noise(current, resistance)
        return noisy_signal

    def apply_all_noise(self, current, resistance):
        # Sequential noise injection
        signal = self.inject_thermal_noise(current, resistance)
        signal = self.inject_shot_noise(signal)
        return signal

    def inject_thermal_noise(self, current, resistance):
        """
        Johnson-Nyquist noise: V_n = sqrt(4 * k_B * T * B * R)
        Injected as current noise: I_n = V_n / R
        """
        # Create noise with the same shape as the input matrix
        noise_std = np.sqrt((4 * self.k_B * self.T * self.B) / resistance)
        noise = np.random.normal(0, noise_std, size=current.shape)
        return current + noise

    def inject_shot_noise(self, current):
        """
        Shot noise: I_n = sqrt(2 * q * I * B)
        """
        # Calculate standard deviation for each point in the matrix
        # Use absolute current to avoid sqrt of negative numbers due to existing noise
        noise_std = np.sqrt(2 * self.q * np.abs(current) * self.B)
        noise = np.random.normal(0, noise_std, size=current.shape)
        return current + noise