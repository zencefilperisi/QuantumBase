# main.py

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Internal module imports
from src.quantum_engine import QuantumBaseMapper
from src.physics_engine import TunnelingPhysics
from src.noise_generator import NoiseGenerator
from src.classifier import BaseCaller
from src.sequencer import NanogapSequencer

app = FastAPI(title="Quantum Nanogap Sequencing API")

# Enable CORS for Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for API ---
class SimulationParams(BaseModel):
    dna_strand: str
    gap_width: float = 1.2    # nanometers
    voltage: float = 0.5      # Volts
    temperature: float = 300  # Kelvin

class StepResult(BaseModel):
    base: str
    theta: float
    phi: float
    current_signal: List[float]

class SimulationResponse(BaseModel):
    original_sequence: str
    predicted_sequence: str
    accuracy: float
    full_signal_trace: List[float]
    quantum_steps: List[StepResult]

# --- Engine Initialization ---
quantum_mapper = QuantumBaseMapper()
noise_gen = NoiseGenerator()
# Note: For real-world use, 'caller' should be pre-trained on a large dataset
caller = BaseCaller() 

@app.get("/health")
async def health_check():
    return {"status": "ready", "engine": "Quantum-Nanogap-V1"}

@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(params: SimulationParams):
    try:
        # 1. Initialize Physics for specific geometry
        physics = TunnelingPhysics(gap_width=params.gap_width, temperature=params.temperature)
        sequencer = NanogapSequencer(physics, noise_gen, caller)
        
        # 2. Execute Sequence processing
        predicted_str, raw_signal = sequencer.sequence_strand(params.dna_strand)
        
        # 3. Generate Step-by-Step Quantum Metadata for Three.js
        quantum_steps = []
        for i, base in enumerate(params.dna_strand):
            q_sig = quantum_mapper.get_quantum_signature(base)
            
            # Extract the specific signal segment for this base
            start_idx = i * 50 # Assuming 50 points per base from sequencer.py
            end_idx = start_idx + 50
            current_segment = raw_signal[start_idx:end_idx].tolist()
            
            step = StepResult(
                base=base,
                theta=q_sig['theta'],
                phi=q_sig['phi'],
                current_signal=current_segment
            )
            quantum_steps.append(step)
            
        # 4. Final Metrics
        correct = sum(1 for a, b in zip(params.dna_strand, predicted_str) if a == b)
        accuracy = (correct / len(params.dna_strand)) * 100

        return SimulationResponse(
            original_sequence=params.dna_strand,
            predicted_sequence=predicted_str,
            accuracy=accuracy,
            full_signal_trace=raw_signal.tolist(),
            quantum_steps=quantum_steps
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Simulation Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")