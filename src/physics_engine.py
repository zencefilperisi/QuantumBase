# src/physics_engine.py_

import numpy as np

class TunnelingPhysics:
    """
    Simulates the quantum tunneling current for DNA nucleotides 
    within a graphene nanogap.
    """
    def __init__(self, gap_width=1.2, temperature=300):
        self.gap_width = gap_width  # Nanometers
        self.temperature = temperature  # Kelvin
        self.voltage = 0.5  # Bias voltage in Volts
        
        # Barrier heights (eV) for DNA bases (Literature values)
        # These represent the electronic signature of each base
        self.barriers = {
            'A': 8.24,
            'T': 9.14,
            'C': 8.90,
            'G': 7.75
        }

    def get_base_current(self, base_type):
        """
        Calculates the tunneling current using the Simmons model approximation.
        I ∝ V * exp(-d * sqrt(Phi))
        """
        phi = self.barriers.get(base_type.upper(), 8.5) # Default barrier
        d = self.gap_width
        
        # Simplified tunneling equation (Quantum mechanics)
        # I = I0 * exp(-2 * d * sqrt(2 * m * Phi) / h_bar)
        decay_constant = 1.025 # eV^-1/2 * A^-1
        
        # Result in nanoAmperes (nA)
        current = self.voltage * np.exp(-decay_constant * d * 10 * np.sqrt(phi))
        
        # Scaling for visibility (to get values in 1-15 nA range)
        return current * 1e8 

    def calculate_boltzmann_noise(self):
        """Returns the thermal noise floor based on temperature."""
        k_b = 1.38e-23
        return np.sqrt(4 * k_b * self.temperature)