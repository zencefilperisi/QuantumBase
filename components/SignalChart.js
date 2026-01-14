// components/SignalChart.js

import dynamic from 'next/dynamic';
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const SignalChart = ({ signalData }) => {
  return (
    <Plot
      data={[
        {
          y: signalData,
          type: 'scatter',
          mode: 'lines',
          marker: { color: '#00f2ff' },
          line: { width: 1 },
          fill: 'tozeroy',
          fillcolor: 'rgba(0, 242, 255, 0.1)'
        },
      ]}
      layout={{
        autosize: true,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { showgrid: false, zeroline: false, visible: false },
        yaxis: { 
          gridcolor: '#334155', 
          title: 'Current (nA)', 
          font: { color: '#fff' },
          range: [0, 15] 
        },
        margin: { t: 20, r: 20, l: 40, b: 20 },
      }}
      style={{ width: '100%', height: '300px' }}
      config={{ displayModeBar: false }}
    />
  );
};

export default SignalChart;