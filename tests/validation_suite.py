# tests/validation_suite.py

import numpy as np
import pandas as pd
from src.physics_engine import TunnelingPhysics
from src.noise_generator import NoiseGenerator
from src.classifier import BaseCaller
from src.data_synthesizer import generate_synthetic_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MonteCarloValidator:
    """
    High-speed validation suite for DNA sequencing accuracy.
    Uses batch processing to accelerate Monte Carlo iterations.
    """
    def __init__(self, iterations=200, strand_length=30):
        self.iterations = iterations
        self.strand_length = strand_length
        self.physics = TunnelingPhysics()
        self.noise_gen = NoiseGenerator()
        self.classifier = BaseCaller()

    def run_validation(self):
        # 1. Training Phase
        print(f"Status: Training classifier with {6000} synthetic samples...")
        training_data = generate_synthetic_dataset(n_samples=6000)
        y_test_train, y_pred_train = self.classifier.train(training_data)

        # 2. High-Speed Batch Testing Phase
        # Total support = iterations * strand_length (e.g., 200 * 30 = 6000)
        total_test_size = self.iterations * self.strand_length
        print(f"Status: Starting Batch Monte Carlo Validation (Total Support: {total_test_size})...")
        
        # Generate all test bases at once
        test_bases = np.random.choice(['A', 'T', 'C', 'G'], total_test_size)
        
        # Vectorized feature generation for testing (Matches synthesizer logic)
        current_map = {b: self.physics.get_base_current(b) for b in ['A', 'T', 'C', 'G']}
        clean_currents = np.array([current_map[b] for b in test_bases])
        
        # Create signal matrix (total_test_size x window_size)
        window_size = 120
        signals = np.tile(clean_currents[:, np.newaxis], (1, window_size))
        noisy_signals = self.noise_gen.add_noise(signals)
        
        # Vectorized feature extraction
        print("Status: Extracting features for 6000 test events...")
        means = np.mean(noisy_signals, axis=1)
        stds = np.std(noisy_signals, axis=1)
        diffs = np.mean(np.abs(np.diff(noisy_signals, axis=1)), axis=1)
        
        fft_vals = np.abs(np.fft.rfft(noisy_signals, axis=1))
        peak_freqs = np.argmax(fft_vals[:, 1:], axis=1)
        
        test_features = pd.DataFrame({
            'mean': means, 'std': stds, 
            'max': np.max(noisy_signals, axis=1), 
            'min': np.min(noisy_signals, axis=1),
            'snr': means / np.where(stds == 0, 1e-12, stds),
            'mean_diff': diffs, 'peak_freq': peak_freqs
        })

        # 3. Batch Prediction
        print("Status: Performing high-speed inference...")
        predictions = self.classifier.model.predict(test_features.values)

        # 4. Reporting
        self.report_results(test_bases, predictions)

    def report_results(self, y_true, y_pred):
        print("\n--- FINAL MONTE CARLO VALIDATION RESULTS ---")
        report = classification_report(y_true, y_pred, target_names=['A', 'C', 'G', 'T'])
        print(report)
        
        # Save results
        if not os.path.exists("results/validation"):
            os.makedirs("results/validation")
            
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=['A', 'C', 'G', 'T'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                    xticklabels=['A', 'C', 'G', 'T'], yticklabels=['A', 'C', 'G', 'T'])
        plt.title(f"Final Validation (Support: {len(y_true)})")
        plt.savefig("results/validation/final_confusion_matrix.png")
        
        # Save CSV report
        report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
        report_df.to_csv("results/validation/final_report.csv")
        print(f"Success: Reports saved to results/validation/")

if __name__ == "__main__":
    validator = MonteCarloValidator(iterations=200, strand_length=30)
    validator.run_validation()