# src/optimizer.py

import numpy as np
import matplotlib.pyplot as plt
from src.data_synthesizer import generate_synthetic_dataset_with_gap # Modifying the synthesizer
from src.classifier import BaseCaller

class GeometryOptimizer:
    """
    Analyzes the impact of electrode gap width on base-calling accuracy
    to determine the optimal device geometry.
    """
    def __init__(self, gap_range=None):
        # Gap widths to test (from 1.0 nm to 2.2 nm)
        self.gap_range = gap_range if gap_range else np.arange(1.0, 2.3, 0.2)
        self.results = []

    def run_optimization_study(self, samples_per_gap=1000):
        """Runs the full pipeline for each gap width."""
        print("--- Starting Geometry Optimization Study ---")
        
        for gap in self.gap_range:
            print(f"Testing Gap Width: {gap:.1f} nm...")
            
            # 1. Generate data for this specific geometry
            # Note: You'll need to update generate_synthetic_dataset to accept gap_width
            data = generate_synthetic_dataset(n_samples=samples_per_gap, gap_width=gap)
            
            # 2. Train and evaluate
            caller = BaseCaller()
            y_true, y_pred = caller.train(data)
            
            # 3. Calculate accuracy
            accuracy = np.mean(y_true == y_pred)
            self.results.append(accuracy)
            
        self.plot_optimization_results()

    def plot_optimization_results(self):
        """Generates the Accuracy vs Gap Width plot for the paper."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.gap_range, self.results, 'o-', color='teal', linewidth=2, markersize=8)
        
        plt.title("Impact of Nanogap Geometry on Sequencing Accuracy", fontsize=14)
        plt.xlabel("Electrode Gap Width (nm)", fontsize=12)
        plt.ylabel("Base-Calling Accuracy (Score)", fontsize=12)
        plt.axvline(x=1.5, color='red', linestyle='--', label='Critical Threshold (1.5nm)')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("results/geometry_optimization.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    optimizer = GeometryOptimizer()
    optimizer.run_optimization_study()