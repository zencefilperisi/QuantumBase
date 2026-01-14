# src/quantum_engine.py

import numpy as np
from typing import Dict

class QuantumBaseMapper:
    """
    Quantum Engine responsible for mapping molecular orbital energies (HOMO-LUMO)
    to Quantum Bloch Sphere coordinates (Theta, Phi).
    """
    def __init__(self):
        # Academic standard values for Ionization Potentials (eV)
        # Reference: "Photoionization of DNA bases", 2024 Synthesis
        self.base_properties = {
            'G': {'energy_ev': 7.75, 'phase_offset': 0.15}, # Guanine (Lowest barrier)
            'A': {'energy_ev': 8.24, 'phase_offset': 0.45}, # Adenine
            'C': {'energy_ev': 8.90, 'phase_offset': 0.75}, # Cytosine
            'T': {'energy_ev': 9.14, 'phase_offset': 1.05}  # Thymine (Highest barrier)
        }
        
        # Physical constants for scaling
        self.energies = [p['energy_ev'] for p in self.base_properties.values()]
        self.min_e, self.max_e = min(self.energies), max(self.energies)

    def get_quantum_signature(self, base_type: str) -> Dict[str, float]:
        """
        Calculates the Bloch Sphere coordinates for a given nucleotide.
        Maps Energy -> Theta (Polar) and Configuration -> Phi (Azimuthal).
        """
        props = self.base_properties.get(base_type.upper(), self.base_properties['A'])
        
        # 1. Theta Calculation (Polar angle [0, PI])
        # Mapping the energy spectrum to the Z-axis rotation of the qubit
        norm_energy = (props['energy_ev'] - self.min_e) / (self.max_e - self.min_e)
        theta = norm_energy * np.pi 
        
        # 2. Phi Calculation (Azimuthal angle [0, 2*PI])
        # Representing molecular configuration phase
        phi = props['phase_offset'] * np.pi
        
        return {
            "theta": float(theta),
            "phi": float(phi),
            "energy_ev": float(props['energy_ev']),
            "base": base_type.upper()
        }

    def batch_process_strand(self, dna_strand: str):
        """Processes an entire DNA sequence into a list of quantum signatures."""
        return [self.get_quantum_signature(base) for base in dna_strand]