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
  const [options, setOptions] = useState({ tracks: [], teams: [], drivers: [], years: [], compounds: [] });
  const [form, setForm] = useState({
    year: 2026,
    track_name: 'Circuit Zandvoort',
    team: 'Red Bull Racing',
    driver: 'VER',
    grid_pos: 1,
    current_lap: 0,
    laps_to_complete: 72,
    // New: Current tire state
    current_compound: '',
    laps_on_current_tire: 0,
    // New: Safety car context
    sc_happened_on_tire: false,
    sc_laps_on_tire: 0,
    sc_currently_out: false,
    // New: Pit history & position
    has_pitted: false,
    track_position: 1,
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
        current_compound: form.current_compound || null,
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

  // Whether we're mid-race (affects which fields are relevant)
  const isMidRace = form.current_lap > 0;

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
        
        {/* === SECTION 1: Race Setup === */}
        <div className="form-section-label">Race Setup</div>
        <div className="form-row">
            <div className="form-group">
                <label>YEAR</label>
                <select id="select-year" value={form.year} onChange={e => setForm({...form, year: parseInt(e.target.value)})}>
                    {options.years.length === 0 ? <option>Loading...</option> : options.years.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TRACK</label>
                <select id="select-track" value={form.track_name} onChange={e => setForm({...form, track_name: e.target.value})}>
                    {options.tracks.length === 0 ? <option>Loading...</option> : options.tracks.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TEAM</label>
                <select id="select-team" value={form.team} onChange={e => setForm({...form, team: e.target.value})}>
                    {options.teams.length === 0 ? <option>Loading...</option> : options.teams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>DRIVER</label>
                <select id="select-driver" value={form.driver} onChange={e => setForm({...form, driver: e.target.value})}>
                    {options.drivers.length === 0 ? <option>Loading...</option> : options.drivers.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
            </div>
        </div>
        
        <div className="form-row" style={{marginTop: '16px'}}>
            <div className="form-group">
                <label>TOTAL RACE LAPS</label>
                <input id="input-total-laps" type="number" min="1" value={form.laps_to_complete} onChange={e => setForm({...form, laps_to_complete: parseInt(e.target.value) || 1})} />
            </div>
            <div className="form-group">
                <label>GRID POSITION</label>
                <input id="input-grid-pos" type="number" min="1" max="20" value={form.grid_pos} onChange={e => setForm({...form, grid_pos: parseInt(e.target.value) || 1})} />
            </div>
        </div>

        {/* === SECTION 2: Race State (Mid-race context) === */}
        <div className="form-section-label" style={{marginTop: '32px'}}>Race State</div>
        <div className="form-row">
            <div className="form-group">
                <label>CURRENT LAP</label>
                <input id="input-current-lap" type="number" min="0" value={form.current_lap} onChange={e => setForm({...form, current_lap: parseInt(e.target.value) || 0})} />
            </div>
            <div className="form-group">
                <label>CURRENT TIRE</label>
                <select id="select-compound" value={form.current_compound} onChange={e => setForm({...form, current_compound: e.target.value})}>
                    <option value="">— Not Set (Pre-Race) —</option>
                    {(options.compounds || ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]).map(c => (
                        <option key={c} value={c}>{c}</option>
                    ))}
                </select>
            </div>
            <div className="form-group">
                <label>LAPS ON CURRENT TIRE</label>
                <input id="input-tire-age" type="number" min="0" value={form.laps_on_current_tire} onChange={e => setForm({...form, laps_on_current_tire: parseInt(e.target.value) || 0})} />
            </div>
            <div className="form-group">
                <label>TRACK POSITION</label>
                <input id="input-track-pos" type="number" min="1" max="20" value={form.track_position} onChange={e => setForm({...form, track_position: parseInt(e.target.value) || 1})} />
            </div>
        </div>

        <div className="form-row" style={{marginTop: '16px'}}>
            <div className="form-group sc-toggle">
                <label>ALREADY PITTED</label>
                <div onClick={() => setForm({...form, has_pitted: !form.has_pitted})}>
                   <input type="checkbox" checked={form.has_pitted} readOnly />
                   <span style={{marginLeft: '10px', fontSize: '0.8rem', fontWeight: 700}}>{form.has_pitted ? 'YES' : 'NO'}</span>
                </div>
            </div>
            <div className="form-group sc-toggle">
                <label>SC ON CURRENT STINT</label>
                <div onClick={() => setForm({...form, sc_happened_on_tire: !form.sc_happened_on_tire})}>
                   <input type="checkbox" checked={form.sc_happened_on_tire} readOnly />
                   <span style={{marginLeft: '10px', fontSize: '0.8rem', fontWeight: 700}}>SC DEPLOYED ON STINT</span>
                </div>
            </div>
            {form.sc_happened_on_tire && (
                <div className="form-group">
                    <label>SC LAPS ON TIRE</label>
                    <input id="input-sc-laps" type="number" min="0" value={form.sc_laps_on_tire} onChange={e => setForm({...form, sc_laps_on_tire: parseInt(e.target.value) || 0})} />
                </div>
            )}
            <div className="form-group sc-toggle">
                <label>SAFETY CAR</label>
                <div onClick={() => setForm({...form, sc_currently_out: !form.sc_currently_out})}>
                   <input type="checkbox" checked={form.sc_currently_out} readOnly />
                   <span style={{marginLeft: '10px', fontSize: '0.8rem', fontWeight: 700}}>CURRENTLY ACTIVE</span>
                </div>
            </div>
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
                    {results.weather_condition && (
                        <span style={{marginLeft: '16px', color: results.weather_condition === 'dry' ? 'var(--text-tertiary)' : 'var(--c-wet)'}}>
                            ● {results.weather_condition.toUpperCase().replace('_', ' ')}
                        </span>
                    )}
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
                            
                            {/* Performance Cliff markers (tire physics — dashed) */}
                            {results.degradation_graphs.SOFT?.performance_cliff_lap && <ReferenceLine x={results.degradation_graphs.SOFT.performance_cliff_lap} stroke="var(--c-soft)" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: 'CLIFF', fill: 'var(--c-soft)', fontSize: 9, fontWeight: 900, position: 'top'}} />}
                            {results.degradation_graphs.MEDIUM?.performance_cliff_lap && <ReferenceLine x={results.degradation_graphs.MEDIUM.performance_cliff_lap} stroke="var(--c-medium)" strokeDasharray="3 3" strokeWidth={1.5} />}
                            {results.degradation_graphs.HARD?.performance_cliff_lap && <ReferenceLine x={results.degradation_graphs.HARD.performance_cliff_lap} stroke="var(--c-hard)" strokeDasharray="3 3" strokeWidth={1.5} />}

                            {/* Strategy Useful Life markers (pit strategy — bold dashed) */}
                            {results.degradation_graphs.SOFT?.strategy_useful_life_lap && <ReferenceLine x={results.degradation_graphs.SOFT.strategy_useful_life_lap} stroke="var(--c-soft)" strokeDasharray="8 4" strokeWidth={2.5} label={{ value: 'PIT', fill: 'var(--c-soft)', fontSize: 9, fontWeight: 900, position: 'insideTopRight'}} />}
                            {results.degradation_graphs.MEDIUM?.strategy_useful_life_lap && <ReferenceLine x={results.degradation_graphs.MEDIUM.strategy_useful_life_lap} stroke="var(--c-medium)" strokeDasharray="8 4" strokeWidth={2.5} />}
                            {results.degradation_graphs.HARD?.strategy_useful_life_lap && <ReferenceLine x={results.degradation_graphs.HARD.strategy_useful_life_lap} stroke="var(--c-hard)" strokeDasharray="8 4" strokeWidth={2.5} />}
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
                                <p><strong>PERF. CLIFF:</strong> <span className="mono" style={{color: '#fff'}}>{data.performance_cliff_lap ? `LAP ${data.performance_cliff_lap}` : 'NONE'}</span></p>
                                <p><strong>USEFUL LIFE:</strong> <span className="mono" style={{color: '#fff'}}>LAP {data.strategy_useful_life_lap}</span></p>
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
                        <div className="strat-philosophy">Mathematically fastest race time based on degradation curves.</div>
                        <StrategyBar strategy={results.strategies.best_strategy} totalLaps={results.strategies.best_strategy?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                        <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.best_strategy?.sequence}</div>
                    </div>

                    <div className="strat-row safe">
                        <div className="strat-title">
                            <h4>SAFE</h4>
                            <span className="mono">{(results.strategies.safe_strategy?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                        </div>
                        <div className="strat-philosophy">Avoids tire cliffs · Prefers wet tires in rain · Low risk of tire failure.</div>
                        <StrategyBar strategy={results.strategies.safe_strategy} totalLaps={results.strategies.safe_strategy?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                        <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.safe_strategy?.sequence}</div>
                    </div>

                    <div className="strat-row risky">
                        <div className="strat-title">
                            <h4>RISKY</h4>
                            <span className="mono">{(results.strategies.risky_strategy?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                        </div>
                        <div className="strat-philosophy">Extends stints past cliff · Gambles on drys in light rain · Pits under SC for faster tires.</div>
                        <StrategyBar strategy={results.strategies.risky_strategy} totalLaps={results.strategies.risky_strategy?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                        <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.risky_strategy?.sequence}</div>
                    </div>
                </div>
            </div>
        </div>
      )}
    </div>
  );
}

export default App;
