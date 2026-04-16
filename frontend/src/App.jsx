import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

function App() {
  const [options, setOptions] = useState({ tracks: [], teams: [], drivers: [], years: [] });
  
  const [form, setForm] = useState({
    year: 2026,
    track_name: 'Circuit Zandvoort',
    team: 'Red Bull Racing',
    driver: 'VER',
    grid_pos: 1,
    current_lap: 0,
    laps_to_complete: 72,
    sc_active: false,
    sc_lap: 10,
    sc_duration: 3
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  // Fetch Dropdown options on boot
  useEffect(() => {
    fetch('/api/options')
      .then(res => res.json())
      .then(data => setOptions(data))
      .catch(err => console.error("API options error:", err));
  }, []);

  const handleSimulate = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Package sc_laps only if active
    const payload = {
        ...form,
        sc_lap: form.sc_active ? form.sc_lap : null,
        sc_duration: form.sc_active ? form.sc_duration : null
    };

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
            setResults(data);
        } else {
            alert('Simulation Error: ' + data.message);
        }
    } catch (err) {
        console.error(err);
    }
    setLoading(false);
  };

  // Restructure the degradation dictionary into Recharts array format
  // graph_data keys are JSON strings ("1", "2", ...) so we must access with String(i)
  const generateChartData = () => {
    if (!results) return [];
    
    const graphs = results.degradation_graphs;
    
    // Find global max lap across all compounds
    let maxLap = 0;
    Object.values(graphs).forEach(c => {
        const keys = Object.keys(c.graph_data).map(Number);
        const localMax = Math.max(...keys);
        if (localMax > maxLap) maxLap = localMax;
    });

    const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];
    
    // Pre-compute last known value per compound for carry-forward
    const lastKnown = {};
    COMPOUNDS.forEach(comp => {
        if (graphs[comp]) {
            const keys = Object.keys(graphs[comp].graph_data).map(Number).sort((a, b) => a - b);
            const lastKey = keys[keys.length - 1];
            lastKnown[comp] = graphs[comp].graph_data[String(lastKey)];
        }
    });

    const formatData = [];
    for (let i = 1; i <= maxLap; i++) {
        let row = { lap: i };
        COMPOUNDS.forEach(comp => {
            if (!graphs[comp]) return;
            const val = graphs[comp].graph_data[String(i)];
            if (val !== undefined) {
                row[comp] = val;
                lastKnown[comp] = val;
            } else {
                // Carry-forward: keep last known lap time so lines extend horizontally
                row[comp] = lastKnown[comp];
            }
        });
        formatData.push(row);
    }
    return formatData;
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>F1 TIRE STRATEGY PREDICTOR</h1>
        <div className="status-badge"><span className="dot"></span>LIVE API DATA</div>
      </header>

      {/* INPUT PANEL */}
      <form className="panel input-panel" onSubmit={handleSimulate}>
        <div className="form-row">
            <div className="form-group">
                <label>YEAR</label>
                <select value={form.year} onChange={e => setForm({...form, year: parseInt(e.target.value)})}>
                    {options.years.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TRACK</label>
                <select value={form.track_name} onChange={e => setForm({...form, track_name: e.target.value})}>
                    {options.tracks.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TEAM</label>
                <select value={form.team} onChange={e => setForm({...form, team: e.target.value})}>
                    {options.teams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>DRIVER</label>
                <select value={form.driver} onChange={e => setForm({...form, driver: e.target.value})}>
                    {options.drivers.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
            </div>
        </div>
        
        <div className="form-row mt-3">
            <div className="form-group">
                <label>CURRENT LAP (0 = Pre-Race)</label>
                <input type="number" min="0" value={form.current_lap} onChange={e => setForm({...form, current_lap: parseInt(e.target.value)})} />
            </div>
            <div className="form-group">
                <label>GRID POSITION</label>
                <input type="number" min="1" max="20" value={form.grid_pos} onChange={e => setForm({...form, grid_pos: parseInt(e.target.value)})} />
            </div>
            <div className="form-group sc-toggle">
                <label>SAFETY CAR ENCOUNTERED?</label>
                <div>
                   <input type="checkbox" checked={form.sc_active} onChange={e => setForm({...form, sc_active: e.target.checked})} />
                   <span style={{marginLeft: '10px'}}>Active</span>
                </div>
            </div>
            {form.sc_active && (
                <>
                <div className="form-group">
                    <label>DEPLOY LAP</label>
                    <input type="number" value={form.sc_lap} onChange={e => setForm({...form, sc_lap: parseInt(e.target.value)})} />
                </div>
                <div className="form-group">
                    <label>DURATION (LAPS)</label>
                    <input type="number" value={form.sc_duration} onChange={e => setForm({...form, sc_duration: parseInt(e.target.value)})} />
                </div>
                </>
            )}
        </div>
        <button type="submit" className="btn-simulate" disabled={loading}>
            {loading ? 'SIMULATING...' : 'RUN STRATEGY SIMULATION'}
        </button>
      </form>

      {/* RESULTS SECTIONS */}
      {results && (
        <>
            <div className="panel chart-panel">
                <h3>TIRE DEGRADATION & PERFORMANCE DROPOFF</h3>
                <div style={{width: '100%', height: '400px'}}>
                    <ResponsiveContainer>
                        <LineChart data={generateChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="lap" type="number" domain={[1, 'dataMax']} stroke="#8E8E93" label={{ value: 'Tire Age (Laps)', position: 'insideBottomRight', offset: -5, fill: '#8E8E93'}} />
                            <YAxis stroke="#8E8E93" domain={['auto', 'auto']} tickFormatter={(v) => `${v.toFixed(1)}s`} label={{ value: 'Lap Time (seconds)', angle: -90, position: 'insideLeft', fill: '#8E8E93' }}/>
                            <Tooltip contentStyle={{backgroundColor: '#1A1A1C', borderColor: '#333'}} />
                            <Legend />
                            {results.degradation_graphs.SOFT && <Line type="monotone" dataKey="SOFT" stroke="var(--c-soft)" strokeWidth={3} dot={false} />}
                            {results.degradation_graphs.MEDIUM && <Line type="monotone" dataKey="MEDIUM" stroke="var(--c-medium)" strokeWidth={3} dot={false} />}
                            {results.degradation_graphs.HARD && <Line type="monotone" dataKey="HARD" stroke="var(--c-hard)" strokeWidth={3} dot={false} />}
                            {results.degradation_graphs.INTERMEDIATE && <Line type="monotone" dataKey="INTERMEDIATE" stroke="var(--c-inter)" strokeWidth={3} dot={false} />}
                            {results.degradation_graphs.WET && <Line type="monotone" dataKey="WET" stroke="var(--c-wet)" strokeWidth={3} dot={false} />}
                            
                            {/* Plot Cliff Point Markers natively for Dry Tires */}
                            {results.degradation_graphs.SOFT && <ReferenceLine x={results.degradation_graphs.SOFT.cliff_point_lap} stroke="var(--c-soft)" strokeDasharray="3 3" />}
                            {results.degradation_graphs.MEDIUM && <ReferenceLine x={results.degradation_graphs.MEDIUM.cliff_point_lap} stroke="var(--c-medium)" strokeDasharray="3 3" />}
                            {results.degradation_graphs.HARD && <ReferenceLine x={results.degradation_graphs.HARD.cliff_point_lap} stroke="var(--c-hard)" strokeDasharray="3 3" />}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                
                <div className="metrics-cards">
                    {["SOFT", "MEDIUM", "HARD"].map(comp => {
                        const data = results.degradation_graphs[comp];
                        if (!data) return null;
                        return (
                            <div className="card" key={comp}>
                                <h4 style={{color: `var(--c-${comp.toLowerCase()})`}}>{comp}</h4>
                                <p><strong>Cliff Point:</strong> Lap {data.cliff_point_lap} (Gradient: {data.cliff_gradient_lap}, Kneedle: {data.cliff_kneedle_lap})</p>
                                <p><strong>Baseline Degradation:</strong> +{data.drop_off_per_lap_sec.toFixed(3)}s / lap</p>
                            </div>
                        )
                    })}
                </div>
            </div>

            <div className="panel strategy-panel">
                <h3>STRATEGY RECOMMENDATIONS</h3>
                
                <div className="strat-row">
                    <div className="strat-title">
                        <h4>BEST STRATEGY</h4>
                        <span>{(results.strategies.best_strategy?.total_optimal_delta || 0).toFixed(1)}s Delta</span>
                    </div>
                    <div className="strat-sequence">{results.strategies.best_strategy?.sequence}</div>
                </div>

                <div className="strat-row safe">
                    <div className="strat-title">
                        <h4>SAFER ALTERNATIVE</h4>
                        <span>{(results.strategies.safer_alternative?.total_optimal_delta || 0).toFixed(1)}s Delta</span>
                    </div>
                    <div className="strat-sequence">{results.strategies.safer_alternative?.sequence}</div>
                </div>

                <div className="strat-row aggressive">
                    <div className="strat-title">
                        <h4>AGGRESSIVE ALTERNATIVE</h4>
                        <span>{(results.strategies.aggressive_alternative?.total_optimal_delta || 0).toFixed(1)}s Delta</span>
                    </div>
                    <div className="strat-sequence">{results.strategies.aggressive_alternative?.sequence}</div>
                </div>
            </div>
        </>
      )}
    </div>
  );
}

export default App;
