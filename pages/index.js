// pages/index.js

import React, { useState } from 'react';
import axios from 'axios';
import { Canvas } from '@react-three/fiber';
import BlochSphere from '../components/BlochSphere';
import SignalChart from '../components/SignalChart';
import { useSequencer } from '../hooks/useSequencer';

export default function Home() {
  const [dnaInput, setDnaInput] = useState("ATGCGT");
  const [simResults, setSimResults] = useState(null);
  const { displayData, isPlaying, setIsPlaying, setCurrentStep } = useSequencer(simResults);

  const startSimulation = async () => {
    const res = await axios.post('http://localhost:8000/simulate', {
      dna_strand: dnaInput,
      gap_width: 1.2
    });
    setSimResults(res.data);
    setCurrentStep(0);
    setIsPlaying(true);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Header */}
      <nav className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 backdrop-blur-md">
        <h1 className="text-xl font-bold tracking-widest text-blue-400">QUANTUM SEQ v1.0</h1>
        <div className="flex gap-4">
          <input 
            className="bg-slate-800 border border-slate-700 px-4 py-2 rounded text-sm uppercase tracking-widest"
            value={dnaInput}
            onChange={(e) => setDnaInput(e.target.value)}
          />
          <button 
            onClick={startSimulation}
            className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded font-bold transition-all"
          >
            {isPlaying ? "SIMULATING..." : "RUN VQE SIMULATION"}
          </button>
        </div>
      </nav>

      <main className="grid grid-cols-12 gap-6 p-8">
        {/* Left: Signal Analysis */}
        <div className="col-span-8 space-y-6">
          <div className="bg-slate-900/80 p-6 rounded-2xl border border-slate-800 shadow-2xl">
            <h3 className="text-sm uppercase text-slate-500 mb-4 tracking-tighter">Real-time Tunneling Current Trace</h3>
            <SignalChart signalData={displayData.signalTrace} />
          </div>
          
          <div className="grid grid-cols-3 gap-6">
            <StatCard label="Base Call" value={displayData.currentBase || "-"} color="text-yellow-400" />
            <StatCard label="Accuracy" value={simResults ? `%${simResults.accuracy.toFixed(1)}` : "-"} color="text-green-400" />
            <StatCard label="System State" value={isPlaying ? "ACTIVE" : "IDLE"} color="text-blue-400" />
          </div>
        </div>

        {/* Right: Quantum Sphere */}
        <div className="col-span-4 bg-black/40 rounded-2xl border border-slate-800 relative overflow-hidden">
          <div className="absolute top-4 left-4 z-10">
            <h3 className="text-xs uppercase text-blue-500 font-bold">Qubit Bloch State</h3>
            <p className="text-[10px] text-slate-500">VQE OPTIMIZED PARAMETERS</p>
          </div>
          <Canvas camera={{ position: [0, 0, 2.5] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <BlochSphere theta={displayData.theta} phi={displayData.phi} />
          </Canvas>
        </div>
      </main>
    </div>
  );
}

const StatCard = ({ label, value, color }) => (
  <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
    <p className="text-[10px] uppercase text-slate-500 mb-1">{label}</p>
    <p className={`text-2xl font-black ${color}`}>{value}</p>
  </div>
);