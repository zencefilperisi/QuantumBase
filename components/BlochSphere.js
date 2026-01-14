// components/BlochSphere

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

const BlochSphere = ({ theta, phi }) => {
  const vectorRef = useRef();

  // Convert spherical coordinates to Cartesian for the Bloch vector
  // x = sin(theta) * cos(phi)
  // y = sin(theta) * sin(phi)
  // z = cos(theta)
  const x = Math.sin(theta) * Math.cos(phi);
  const y = Math.sin(theta) * Math.sin(phi);
  const z = Math.cos(theta);

  return (
    <group>
      {/* Main Sphere - Translucent wireframe for the Bloch Sphere */}
      <Sphere args={[1, 32, 32]}>
        <meshStandardMaterial color="#30415d" wireframe transparent opacity={0.3} />
      </Sphere>

      {/* Axis Lines: X, Y, Z */}
      <Line points={[[-1.2, 0, 0], [1.2, 0, 0]]} color="gray" lineWidth={1} /> {/* X-axis */}
      <Line points={[[0, -1.2, 0], [0, 1.2, 0]]} color="gray" lineWidth={1} /> {/* Y-axis */}
      <Line points={[[0, 0, -1.2], [0, 0, 1.2]]} color="gray" lineWidth={1} /> {/* Z-axis */}

      {/* Labels for States */}
      <Html position={[0, 1.3, 0]}><span className="text-white text-xs">|0⟩</span></Html>
      <Html position={[0, -1.3, 0]}><span className="text-white text-xs">|1⟩</span></Html>

      {/* The Bloch Vector representing the current Base Signature */}
      <Line 
        points={[[0, 0, 0], [x, z, y]]} // Three.js uses Y as up, so we swap Z and Y
        color="#ffcc00" 
        lineWidth={3} 
      />
      
      {/* Vector Tip (Head) */}
      <mesh position={[x, z, y]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial color="#ffcc00" emissive="#ffcc00" />
      </mesh>
    </group>
  );
};

export default BlochSphere;