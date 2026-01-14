# src/fast5_extractor.py

import os
import numpy as np
import pandas as pd
from ont_fast5_api.fast5_interface import get_fast5_file

class Fast5SignalExtractor:
    """
    A professional utility to extract raw ionic current signals from 
    Oxford Nanopore .fast5 files for Bi-LSTM training and inference.
    """
    def __init__(self, window_size=30):
        self.window_size = window_size

    def process_directory(self, input_dir, output_csv):
        """
        Scans a directory for .fast5 files and extracts signal chunks.
        """
        all_signals = []
        
        if not os.path.exists(input_dir):
            print(f"[Error] Directory not found: {input_dir}")
            return

        print(f"[System] Initiating signal extraction from: {input_dir}")
        
        fast5_files = [f for f in os.listdir(input_dir) if f.endswith(".fast5")]
        
        for filename in fast5_files:
            file_path = os.path.join(input_dir, filename)
            
            try:
                with get_fast5_file(file_path, mode="r") as f5:
                    for read in f5.get_reads():
                        # Extracting raw pA (picoampere) signals
                        raw_signal = read.get_raw_data()
                        
                        # Data Segmentation: Breaking the long signal into chunks
                        # that match our Bi-LSTM input dimension (e.g., 30 samples)
                        num_chunks = len(raw_signal) // self.window_size
                        reshaped_signal = raw_signal[:num_chunks * self.window_size].reshape(-1, self.window_size)
                        
                        for chunk in reshaped_signal:
                            # In a real-world supervised learning scenario, 
                            # labels are obtained via basecall-alignment (Event Detection).
                            # Here, we provide a placeholder label (0-3) for structural integrity.
                            mock_label = np.random.randint(0, 4) 
                            
                            # Append signal features + target label
                            combined_row = np.append(chunk, mock_label)
                            all_signals.append(combined_row)
                            
            except Exception as e:
                print(f"[Skipping] Error processing {filename}: {str(e)}")

        # Saving to CSV for train_model.py
        if all_signals:
            df = pd.DataFrame(all_signals)
            # Naming columns: Signal_0, Signal_1 ... Label
            columns = [f"Signal_{i}" for i in range(self.window_size)] + ["Label"]
            df.columns = columns
            
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"[Success] Extraction complete. {len(all_signals)} samples saved to {output_csv}")
        else:
            print("[Warning] No signals were extracted. Check your .fast5 file integrity.")

if __name__ == "__main__":
    # Setup paths
    RAW_DATA_PATH = "data/raw_fast5"
    OUTPUT_FILE = "data/genomic_signals.csv"
    
    # Initialize and Run
    extractor = Fast5SignalExtractor(window_size=30)
    extractor.process_directory(RAW_DATA_PATH, OUTPUT_FILE)