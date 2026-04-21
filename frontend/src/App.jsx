import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip panel mono" style={{ padding: '12px', border: '1px solid var(--f1-red)', borderLeft: '4px solid var(--f1-red)' }}>
        <p className="label" style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{ `LAP ${label}` }</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ margin: '4px 0', color: entry.color, fontWeight: 'bold' }}>
            {entry.name}: {entry.value.toFixed(3)}s
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const StrategyBar = ({ strategy, totalLaps }) => {
  if (!strategy || !strategy.stints_data) return null;
  
  return (
    <div className="stint-bar-container">
      {strategy.stints_data.map((stint, idx) => (
        <div 
          key={idx} 
          className="stint-segment" 
          data-compound={stint.compound}
          style={{ width: `${(stint.laps / totalLaps) * 100}%` }}
          title={`${stint.compound}: Lap ${stint.start} - ${stint.end} (${stint.laps} laps)`}
        >
          <span className="stint-label">{stint.compound}</span>
          <span className="stint-laps">{stint.laps}</span>
        </div>
      ))}
    </div>
  );
};

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

  useEffect(() => {
    fetch('/api/options')
      .then(res => res.json())
      .then(data => setOptions(data))
      .catch(err => console.error("API options error:", err));
  }, []);

  const handleSimulate = async (e) => {
    e.preventDefault();
    setLoading(true);
    
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

  const generateChartData = () => {
    if (!results) return [];
    const graphs = results.degradation_graphs;
    let maxLap = 0;
    Object.values(graphs).forEach(c => {
        const keys = Object.keys(c.graph_data).map(Number);
        const localMax = Math.max(...keys);
        if (localMax > maxLap) maxLap = localMax;
    });

    const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];
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
        <h1>F1 TIRE STRATEGY <span style={{color: 'var(--text-secondary)', fontWeight: 300}}>PREDICTOR</span></h1>
        <div className="status-badge"><span className="dot"></span> LIVE TELEMETRY FEED</div>
      </header>

      <form className="panel input-panel" onSubmit={handleSimulate}>
        {loading && (
            <div className="calc-animation">
                <h3 style={{fontStyle: 'italic', color: 'var(--text-primary)'}}>PROCESSING STRATEGY...</h3>
                <div className="loading-bar-container">
                    <div className="loading-bar"></div>
                </div>
            </div>
        )}
        
        <div className="form-row">
            <div className="form-group">
                <label>YEAR</label>
                <select value={form.year} onChange={e => setForm({...form, year: parseInt(e.target.value)})}>
                    {options.years.length === 0 ? <option>Loading...</option> : options.years.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TRACK</label>
                <select value={form.track_name} onChange={e => setForm({...form, track_name: e.target.value})}>
                    {options.tracks.length === 0 ? <option>Loading...</option> : options.tracks.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TEAM</label>
                <select value={form.team} onChange={e => setForm({...form, team: e.target.value})}>
                    {options.teams.length === 0 ? <option>Loading...</option> : options.teams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>DRIVER</label>
                <select value={form.driver} onChange={e => setForm({...form, driver: e.target.value})}>
                    {options.drivers.length === 0 ? <option>Loading...</option> : options.drivers.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
            </div>
        </div>
        
        <div className="form-row mt-3" style={{marginTop: '24px'}}>
            <div className="form-group">
                <label>CURRENT LAP</label>
                <input type="number" min="0" value={form.current_lap} onChange={e => setForm({...form, current_lap: parseInt(e.target.value)})} />
            </div>
            <div className="form-group">
                <label>GRID POSITION</label>
                <input type="number" min="1" max="20" value={form.grid_pos} onChange={e => setForm({...form, grid_pos: parseInt(e.target.value)})} />
            </div>
            <div className="form-group sc-toggle">
                <label>SAFETY CAR</label>
                <div onClick={() => setForm({...form, sc_active: !form.sc_active})}>
                   <input type="checkbox" checked={form.sc_active} readOnly />
                   <span style={{marginLeft: '10px', fontSize: '0.8rem', fontWeight: 700}}>ACTIVE DEPLOYMENT</span>
                </div>
            </div>
            {form.sc_active && (
                <>
                <div className="form-group">
                    <label>DEPLOY LAP</label>
                    <input type="number" value={form.sc_lap} onChange={e => setForm({...form, sc_lap: parseInt(e.target.value)})} />
                </div>
                <div className="form-group">
                    <label>DURATION</label>
                    <input type="number" value={form.sc_duration} onChange={e => setForm({...form, sc_duration: parseInt(e.target.value)})} />
                </div>
                </>
            )}
        </div>
        <button type="submit" className="btn-simulate" disabled={loading}>
            {loading ? 'CALCULATING OPTIMAL PATH...' : 'RUN STRATEGY SIMULATION'}
        </button>
      </form>

      {results && !loading && (
        <div className="fade-in-up">
            <div className="results-header">
                <h3>TIRE DEGRADATION PROFILE</h3>
                <div className="mono" style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
                    SYNCED: {new Date().toLocaleTimeString()}
                </div>
            </div>

            <div className="chart-container">
                <div style={{width: '100%', height: '450px'}}>
                    <ResponsiveContainer>
                        <LineChart data={generateChartData()} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2B2B35" vertical={false} />
                            <XAxis 
                                dataKey="lap" 
                                type="number" 
                                domain={[1, 'dataMax']} 
                                stroke="var(--text-tertiary)" 
                                tick={{fontFamily: 'JetBrains Mono', fontSize: 10}}
                                label={{ value: 'TIRE AGE (LAPS)', position: 'insideBottom', offset: -10, fill: 'var(--text-secondary)', style: {fontSize: 10, fontWeight: 700}}} 
                            />
                            <YAxis 
                                stroke="var(--text-tertiary)" 
                                domain={['auto', 'auto']} 
                                tick={{fontFamily: 'JetBrains Mono', fontSize: 10}}
                                tickFormatter={(v) => `${v.toFixed(1)}s`} 
                                label={{ value: 'LAP TIME DELTA', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', style: {fontSize: 10, fontWeight: 700} }}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend wrapperStyle={{paddingTop: '20px', fontFamily: 'Outfit', fontWeight: 900, fontSize: '10px'}} />
                            
                            {results.degradation_graphs.SOFT && <Line type="monotone" dataKey="SOFT" stroke="var(--c-soft)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                            {results.degradation_graphs.MEDIUM && <Line type="monotone" dataKey="MEDIUM" stroke="var(--c-medium)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                            {results.degradation_graphs.HARD && <Line type="monotone" dataKey="HARD" stroke="var(--c-hard)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                            {results.degradation_graphs.INTERMEDIATE && <Line type="monotone" dataKey="INTERMEDIATE" stroke="var(--c-inter)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                            {results.degradation_graphs.WET && <Line type="monotone" dataKey="WET" stroke="var(--c-wet)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                            
                            {results.degradation_graphs.SOFT && <ReferenceLine x={results.degradation_graphs.SOFT.cliff_point_lap} stroke="var(--c-soft)" strokeDasharray="5 5" label={{ value: 'CLIFF', fill: 'var(--c-soft)', fontSize: 10, fontWeight: 900, position: 'top'}} />}
                            {results.degradation_graphs.MEDIUM && <ReferenceLine x={results.degradation_graphs.MEDIUM.cliff_point_lap} stroke="var(--c-medium)" strokeDasharray="5 5" />}
                            {results.degradation_graphs.HARD && <ReferenceLine x={results.degradation_graphs.HARD.cliff_point_lap} stroke="var(--c-hard)" strokeDasharray="5 5" />}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                
                <div className="metrics-cards">
                    {["SOFT", "MEDIUM", "HARD"].map(comp => {
                        const data = results.degradation_graphs[comp];
                        if (!data) return null;
                        return (
                            <div className="card" key={comp} style={{borderLeftColor: `var(--c-${comp.toLowerCase()})`}}>
                                <h4 style={{color: `var(--c-${comp.toLowerCase()})`}}>COMPOUND: {comp}</h4>
                                <p><strong>CLIFF POINT:</strong> LAP <span className="mono" style={{color: '#fff'}}>{data.cliff_point_lap}</span></p>
                                <p><strong>GRADIENT:</strong> <span className="mono" style={{color: '#fff'}}>{data.cliff_gradient_lap}</span></p>
                                <p><strong>EST. DEGRADATION:</strong> <span className="mono" style={{color: '#fff'}}>+{data.drop_off_per_lap_sec.toFixed(3)}s</span> / lap</p>
                            </div>
                        )
                    })}
                </div>
            </div>

            <div className="panel strategy-panel">
                <h3 style={{marginBottom: '24px'}}>STRATEGY RECOMMENDATIONS</h3>
                
                <div className="strategy-grid">
                    <div className="strat-row best">
                        <div className="strat-title">
                            <h4>OPTIMAL STRATEGY</h4>
                            <span className="mono">{(results.strategies.best_strategy?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                        </div>
                        <StrategyBar strategy={results.strategies.best_strategy} totalLaps={results.strategies.best_strategy?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                        <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.best_strategy?.sequence}</div>
                    </div>

                    <div className="strat-row safe">
                        <div className="strat-title">
                            <h4>CONSERVATIVE</h4>
                            <span className="mono">{(results.strategies.safer_alternative?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                        </div>
                        <StrategyBar strategy={results.strategies.safer_alternative} totalLaps={results.strategies.safer_alternative?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                        <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.safer_alternative?.sequence}</div>
                    </div>

                    <div className="strat-row aggressive">
                        <div className="strat-title">
                            <h4>AGGRESSIVE</h4>
                            <span className="mono">{(results.strategies.aggressive_alternative?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                        </div>
                        <StrategyBar strategy={results.strategies.aggressive_alternative} totalLaps={results.strategies.aggressive_alternative?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                        <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.aggressive_alternative?.sequence}</div>
                    </div>
                </div>
            </div>
        </div>
      )}
    </div>
  );
}

export default App;
