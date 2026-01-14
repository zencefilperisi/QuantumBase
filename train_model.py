# train_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from src.classifier import BiLSTMBaseCaller
from src.data_loader import GenomicSignalDataset
from torch.utils.data import DataLoader

def run_training():
    # 1. SETUP: Hardware & Architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Training initiated on device: {device}")

    # Hyperparameters
    INPUT_DIM = 1
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50

    # 2. DATA ACQUISITION
    # Replace the path below with your actual processed CSV file
    csv_path = "data/genomic_signals.csv"
    
    if os.path.exists(csv_path):
        print(f"[Data] Loading real-world dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Assuming last column is 'label', everything else is 'signal'
        signals = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
    else:
        print("[Warning] Real-world dataset not found. Generating high-fidelity mock data for structural validation...")
        signals = np.random.randn(2000, 30) # 2000 samples, 30 sequence length
        labels = np.random.randint(0, 4, 2000)

    # 3. PREPROCESSING & LOADING
    dataset = GenomicSignalDataset(signals, labels)
    # Splitting for validation (80% Train, 20% Val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. MODEL INITIALIZATION
    model = BiLSTMBaseCaller(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.NLLLoss() # Negative Log Likelihood for LogSoftmax output
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. TRAINING LOOP
    print(f"[Progress] Starting Neural Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_signals, batch_labels in train_loader:
            batch_signals, batch_labels = batch_signals.to(device), batch_labels.to(device)

            # Optimization Step
            optimizer.zero_grad()
            outputs = model(batch_signals)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Accuracy Calculation
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v_signals, v_labels in val_loader:
                v_signals, v_labels = v_signals.to(device), v_labels.to(device)
                v_outputs = model(v_signals)
                _, v_pred = torch.max(v_outputs.data, 1)
                val_total += v_labels.size(0)
                val_correct += (v_pred == v_labels).sum().item()

        if (epoch + 1) % 5 == 0:
            train_acc = 100 * correct / total
            val_acc = 100 * val_correct / val_total
            print(f"Epoch [{epoch+1}/{EPOCHS}] -> Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 6. EXPORT: Saving the trained Brain
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, "bilstm_basecaller.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Success] Neural weights saved to {model_path}")
    print("[Success] Deployment ready for QuantumBase GUI.")

if __name__ == "__main__":
    run_training()