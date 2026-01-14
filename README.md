# QuantumBase  
### Real-Time DNA Sequencing & Quantum Tunneling Simulator

QuantumBase Pro is an advanced bioinformatics simulation suite designed to model the identification of DNA nucleotides (A, C, G, T) based on **quantum tunneling current** modulations. Built with Python and PyQt6, it bridges the gap between quantum physics and genomic data science.



## Key Features

- **Physics-Driven Inference:** Simulates ionic flux and tunneling currents using a stochastic model based on the Schrödinger equation principles.
- **Neural Engine Analytics:** Provides deep-dive performance metrics, including confusion matrices and learning curves.
- **High-Noise Processing:** Capable of decoding signals under realistic biological noise conditions (up to 42% variance).
- **Academic Reporting:** Automatically generates high-fidelity PDF reports of sequencing results and quality (Q-Scores).
- **External Data Support:** Interface designed to import and process external signal datasets (Fast5/CSV).

## Technical Stack

- **GUI Framework:** PyQt6 (High-DPI Support)
- **Data Science:** NumPy, Pandas, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Deep Learning Architecture:** Modular design prepared for PyTorch/TensorFlow integration.
- **Reporting:** ReportLab PDF Engine



## Scientific Foundation

The core engine calculates the probability of a nucleotide passing through a 1.2nm nanogap by modeling the barrier height (eV) of each base:
- **Guanine (G):** ~22.5 pA (Lowest barrier, highest conductance)
- **Adenine (A):** ~15.2 pA
- **Cytosine (C):** ~8.4 pA
- **Thymine (T):** ~5.1 pA (Highest barrier, lowest conductance)

The simulator uses a stochastic Gaussian model to account for molecular orientation and thermal noise, providing a realistic Phred Quality Score ($Q = -10 \log_{10} P_{err}$) for every identified base.

## Installation & Usage

1. **Clone the repository:**
```bash
   git clone [https://github.com/zencefilperisi/QuantumBase.git](https://github.com/zencefilperisi/QuantumBase.git)
```

2. **Install Dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Launch the Application:**
```bash
python main_gui.py
```
# Analytics Dashboard

The Performance Suite offers:
- Confusion Matrix: Real-time tracking of True Positives vs. Miscalls.
- Neural Dynamics: Monitoring the convergence of loss and accuracy during the session.

Developed for research and educational purposes in the field of Nanopore Sequencing.