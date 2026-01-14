// hooks/useSequnecer.js

import { useState, useEffect } from 'react';

export const useSequencer = (simulationData) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [displayData, setDisplayData] = useState({
    theta: 0,
    phi: 0,
    currentBase: '',
    signalTrace: []
  });

  useEffect(() => {
    let interval;
    if (isPlaying && simulationData && currentStep < simulationData.quantum_steps.length) {
      interval = setInterval(() => {
        const step = simulationData.quantum_steps[currentStep];
        
        setDisplayData({
          theta: step.theta,
          phi: step.phi,
          currentBase: step.base,
          signalTrace: (prev) => [...prev, ...step.current_signal].slice(-500) 
        });

        setCurrentStep((prev) => prev + 1);
      }, 800); 
    } else {
      setIsPlaying(false);
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentStep, simulationData]);

  return { displayData, isPlaying, setIsPlaying, currentStep, setCurrentStep };
};