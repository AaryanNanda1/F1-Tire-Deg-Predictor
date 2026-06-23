import React, { useState, useEffect } from 'react';
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

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

const WeatherForecast = ({ forecast }) => {
  if (!forecast) return null;
  const { air_temp, track_temp, humidity, rainfall, wind_speed, synopsis, hourly_forecasts } = forecast;

  return (
    <div className="panel weather-panel" style={{ marginBottom: '24px' }}>
        <h3 style={{marginBottom: '16px'}}>RACE WEATHER CONDITIONS</h3>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <div>
                <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{synopsis.toUpperCase()}</p>
                <p className="mono" style={{ margin: '4px 0 0 0', color: 'var(--text-secondary)' }}>
                    AIR: {air_temp}°C | TRACK: {track_temp}°C | HUMIDITY: {humidity}% | WIND: {wind_speed}km/h
                </p>
            </div>
            {rainfall && <div className="mono" style={{ color: 'var(--c-wet)', fontWeight: 'bold' }}>RAIN EXPECTED</div>}
        </div>
        {hourly_forecasts && hourly_forecasts.length > 0 && (
            <div style={{ display: 'flex', gap: '12px' }}>
                {hourly_forecasts.map((hf, i) => (
                    <div key={i} className="card mono" style={{ flex: 1, textAlign: 'center', padding: '12px', borderLeftColor: hf.rainfall_mm > 0 ? 'var(--c-wet)' : 'var(--text-tertiary)' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '8px' }}>{hf.hour}:00</div>
                        <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: hf.rainfall_mm > 0 ? 'var(--c-wet)' : 'var(--text-primary)' }}>
                            {hf.rainfall_mm > 0 ? `${hf.rainfall_mm}mm` : '0.0mm'}
                        </div>
                        <div style={{ fontSize: '0.7rem', color: hf.rainfall_mm > 0 ? 'var(--c-wet)' : 'var(--text-secondary)', marginBottom: '8px', letterSpacing: '1px' }}>PRECIPITATION</div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{hf.track_temp || hf.air_temp}°C <span style={{fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 'normal'}}>TRACK</span></div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '2px' }}>{hf.air_temp}°C <span style={{fontSize: '0.7rem', fontWeight: 'normal'}}>AIR</span></div>
                    </div>
                ))}
            </div>
        )}
    </div>
  );
};

const TrackMetricsChart = ({ features }) => {
  if (!features) return null;
  
  const data = [
    { subject: 'TRACTION', A: (features.traction || 0.5) * 100 },
    { subject: 'LOAD (HS)', A: (features.high_speed_load || 0.5) * 100 },
    { subject: 'ABRASIVE', A: (features.abrasiveness || 0.5) * 100 },
    { subject: 'ROUGHNESS', A: (features.surface_roughness || 0.5) * 100 },
    { subject: 'BRAKING', A: (features.braking_severity || 0.5) * 100 },
    { subject: 'LATERAL', A: (features.lateral_load || 0.5) * 100 },
    { subject: 'TEMP SENS', A: (features.track_temp_sensitivity || 0.5) * 100 },
  ];

  return (
    <div className="panel track-metrics-panel" style={{ flex: 1, minWidth: '350px' }}>
      <h3 style={{ marginBottom: '20px' }}>CIRCUIT CHARACTERISTICS</h3>
      <div style={{ width: '100%', height: '350px' }}>
        <ResponsiveContainer>
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
            <PolarGrid stroke="var(--surface-border)" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 9, fontWeight: 700 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
            <Radar
              name="Metrics"
              dataKey="A"
              stroke="var(--f1-red)"
              fill="var(--f1-red)"
              fillOpacity={0.6}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="track-stress-metrics">
          <div className="stress-item">
              <span className="label">THERMAL STRESS</span>
              <div className="bar-bg"><div className="bar-fill" style={{ width: `${(features.thermal_stress || 0.5) * 100}%` }}></div></div>
          </div>
          <div className="stress-item">
              <span className="label">SURFACE WEAR</span>
              <div className="bar-bg"><div className="bar-fill" style={{ width: `${(features.surface_wear || 0.5) * 100}%` }}></div></div>
          </div>
          <div className="stress-item">
              <span className="label">ENERGY LOAD</span>
              <div className="bar-bg"><div className="bar-fill" style={{ width: `${(features.energy_load || 0.5) * 100}%` }}></div></div>
          </div>
      </div>
    </div>
  );
};


function App() {
  const [options, setOptions] = useState({ tracks: [], teams: [], drivers: [], years: [], compounds: [] });
  const [form, setForm] = useState({
    year: 2026,
    track_name: 'Circuit Zandvoort (Netherlands)',
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
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/options')
      .then(res => res.json())
      .then(data => setOptions(data))
      .catch(err => console.error("API options error:", err));
  }, []);

  const handleSimulate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    const payload = {
        ...form,
        grid_pos: parseInt(form.grid_pos) || 1,
        current_lap: parseInt(form.current_lap) || 0,
        laps_to_complete: parseInt(form.laps_to_complete) || 72,
        laps_on_current_tire: parseInt(form.laps_on_current_tire) || 0,
        sc_laps_on_tire: parseInt(form.sc_laps_on_tire) || 0,
        track_position: parseInt(form.track_position) || 1,
        current_compound: form.current_compound || null,
    };

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const responseText = await res.text();
        let data;
        try {
            data = responseText ? JSON.parse(responseText) : {};
        } catch {
            data = {};
        }

        if (!res.ok) {
            throw new Error(data.message || `Simulation request failed (${res.status})`);
        }
        
        if (data.status === 'success') {
            setResults(data);
        } else {
            throw new Error(data.message || 'The simulation did not return results.');
        }
    } catch (err) {
        console.error(err);
        setError(err.message || 'Unable to run the simulation. Please try again.');
    } finally {
        setLoading(false);
    }
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

  useEffect(() => {
    // Team & Driver validation
    if (options.driver_roster && options.driver_roster[form.year]) {
        const validTeamsForYear = Object.keys(options.driver_roster[form.year]);
        
        if (form.team && !validTeamsForYear.includes(form.team)) {
            setForm(prev => ({ ...prev, team: '', driver: '' }));
        } else if (form.team && options.driver_roster[form.year][form.team]) {
            const teamRoster = options.driver_roster[form.year][form.team];
            if (teamRoster.length > 0 && !teamRoster.includes(form.driver)) {
                setForm(prev => ({ ...prev, driver: teamRoster[0] }));
            }
        }
    }

    // Track validation
    if (options.yearly_tracks && options.yearly_tracks[form.year]) {
        const validTracksForYear = options.yearly_tracks[form.year];
        if (form.track_name && !validTracksForYear.includes(form.track_name)) {
            setForm(prev => ({ ...prev, track_name: '' }));
        }
    }
  }, [form.year, form.team, form.track_name, options.driver_roster, options.yearly_tracks]);

  let validTracks = options.tracks;
  if (options.yearly_tracks && options.yearly_tracks[form.year]) {
      validTracks = options.yearly_tracks[form.year];
  }

  let validTeams = options.teams;
  if (options.driver_roster && options.driver_roster[form.year]) {
      validTeams = Object.keys(options.driver_roster[form.year]);
  }

  let validDrivers = options.drivers;
  if (form.team && options.driver_roster && options.driver_roster[form.year] && options.driver_roster[form.year][form.team]) {
      validDrivers = options.driver_roster[form.year][form.team];
  } else if (!form.team) {
      validDrivers = []; // Clear drivers if no team selected
  }

  const getModelTrainedDate = () => {
    if (!options.model_metadata) return 'LOADING...';
    const eraKey = form.year >= 2026 ? 'active_aero_2026_2030' : 'ground_effect_2022_2025';
    const eraInfo = options.model_metadata[eraKey];
    if (!eraInfo) return 'UNKNOWN';

    const trainedDate = eraInfo.trained_at || eraInfo.as_of;
    if (!trainedDate) return 'UNKNOWN';

    const parsed = new Date(trainedDate);
    return Number.isNaN(parsed.getTime())
        ? trainedDate
        : parsed.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>F1 TIRE STRATEGY <span style={{color: 'var(--text-secondary)', fontWeight: 300}}>PREDICTOR</span></h1>
        <div className="status-badge"><span className="dot"></span> MODEL TRAINED: {getModelTrainedDate()}</div>
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
                <select id="select-track" value={form.track_name} onChange={e => {
                    const newTrack = e.target.value;
                    let defaultLaps = form.laps_to_complete;
                    if (options.track_laps && options.track_laps[newTrack]) {
                        defaultLaps = options.track_laps[newTrack];
                    }
                    setForm({...form, track_name: newTrack, laps_to_complete: defaultLaps});
                }}>
                    <option value="">— Select Track —</option>
                    {validTracks.length === 0 ? <option disabled>Loading...</option> : validTracks.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>TEAM</label>
                <select id="select-team" value={form.team} onChange={e => setForm({...form, team: e.target.value})}>
                    <option value="">— Select Team —</option>
                    {validTeams.length === 0 ? <option disabled>Loading...</option> : validTeams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
            </div>
            <div className="form-group">
                <label>DRIVER</label>
                <select id="select-driver" value={form.driver} onChange={e => setForm({...form, driver: e.target.value})} disabled={!form.team}>
                    {!form.team ? <option value="">— Select Team First —</option> : validDrivers.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
            </div>
        </div>
        
        <div className="form-row" style={{marginTop: '16px'}}>
            <div className="form-group">
                <label>TOTAL RACE LAPS</label>
                <input 
                    id="input-total-laps" 
                    type="text" 
                    inputMode="numeric"
                    value={form.laps_to_complete} 
                    onChange={e => setForm({...form, laps_to_complete: e.target.value.replace(/\D/g, '')})} 
                />
            </div>
            <div className="form-group">
                <label>GRID POSITION</label>
                <input 
                    id="input-grid-pos" 
                    type="text" 
                    inputMode="numeric"
                    value={form.grid_pos} 
                    onChange={e => setForm({...form, grid_pos: e.target.value.replace(/\D/g, '')})} 
                />
            </div>
        </div>

        {/* === SECTION 2: Race State (Mid-race context) === */}
        <div className="form-section-label" style={{marginTop: '32px'}}>Race State</div>
        <div className="form-row">
            <div className="form-group">
                <label>CURRENT LAP</label>
                <input 
                    id="input-current-lap" 
                    type="text" 
                    inputMode="numeric"
                    value={form.current_lap} 
                    onChange={e => setForm({...form, current_lap: e.target.value.replace(/\D/g, '')})} 
                />
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
                <input 
                    id="input-tire-age" 
                    type="text" 
                    inputMode="numeric"
                    value={form.laps_on_current_tire} 
                    onChange={e => setForm({...form, laps_on_current_tire: e.target.value.replace(/\D/g, '')})} 
                />
            </div>
            <div className="form-group">
                <label>TRACK POSITION</label>
                <input 
                    id="input-track-pos" 
                    type="text" 
                    inputMode="numeric"
                    value={form.track_position} 
                    onChange={e => setForm({...form, track_position: e.target.value.replace(/\D/g, '')})} 
                />
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
                    <input 
                        id="input-sc-laps" 
                        type="text" 
                        inputMode="numeric"
                        value={form.sc_laps_on_tire} 
                        onChange={e => setForm({...form, sc_laps_on_tire: e.target.value.replace(/\D/g, '')})} 
                    />
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

      {error && (
        <div className="panel" role="alert" style={{borderColor: 'var(--c-soft)', color: 'var(--c-soft)', marginBottom: '24px'}}>
            <strong>SIMULATION ERROR:</strong> {error}
        </div>
      )}

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
            
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', marginBottom: '32px', alignItems: 'stretch' }}>
                <div style={{ flex: 2, minWidth: '600px' }}>
                    <div className="chart-container" style={{ height: '100%', marginBottom: 0 }}>
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
                                        label={{ value: 'TOTAL LAP TIME', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', style: {fontSize: 10, fontWeight: 700} }}
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Legend wrapperStyle={{paddingTop: '20px', fontFamily: 'Outfit', fontWeight: 900, fontSize: '10px'}} />
                                    
                                    {results.degradation_graphs.SOFT && <Line type="monotone" dataKey="SOFT" stroke="var(--c-soft)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                                    {results.degradation_graphs.MEDIUM && <Line type="monotone" dataKey="MEDIUM" stroke="var(--c-medium)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                                    {results.degradation_graphs.HARD && <Line type="monotone" dataKey="HARD" stroke="var(--c-hard)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                                    {results.degradation_graphs.INTERMEDIATE && <Line type="monotone" dataKey="INTERMEDIATE" stroke="var(--c-inter)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                                    {results.degradation_graphs.WET && <Line type="monotone" dataKey="WET" stroke="var(--c-wet)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
                
                <TrackMetricsChart features={results.input_context?.track_features} />
            </div>

            <div className="metrics-cards" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
                {["SOFT", "MEDIUM", "HARD"].map(comp => {
                    const data = results.degradation_graphs[comp];
                    if (!data) return null;
                    return (
                        <div className="card" key={comp} style={{borderLeftColor: `var(--c-${comp.toLowerCase()})`, margin: 0}}>
                            <h4 style={{color: `var(--c-${comp.toLowerCase()})`, fontSize: '1rem'}}>COMPOUND: {comp}</h4>
                            <p><strong>PERF. CLIFF:</strong> <span className="mono" style={{color: '#fff'}}>{data.performance_cliff_lap ? `LAP ${data.performance_cliff_lap}` : 'NONE'}</span></p>
                            <p><strong>USEFUL LIFE:</strong> <span className="mono" style={{color: '#fff'}}>LAP {data.strategy_useful_life_lap}</span></p>
                            <p><strong>EST. DEGRADATION:</strong> <span className="mono" style={{color: '#fff'}}>+{data.drop_off_per_lap_sec.toFixed(3)}s</span> / lap</p>
                        </div>
                    )
                })}
            </div>
            
            <WeatherForecast forecast={results.weather_forecast} />

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

                    {results.strategies.safe_strategy && (
                        <div className="strat-row safe">
                            <div className="strat-title">
                                <h4>SAFE</h4>
                                <span className="mono">{(results.strategies.safe_strategy?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                            </div>
                            <div className="strat-philosophy">Avoids tire cliffs · Prefers wet tires in rain · Low risk of tire failure.</div>
                            <StrategyBar strategy={results.strategies.safe_strategy} totalLaps={results.strategies.safe_strategy?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                            <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.safe_strategy?.sequence}</div>
                        </div>
                    )}

                    {results.strategies.risky_strategy && (
                        <div className="strat-row risky">
                            <div className="strat-title">
                                <h4>RISKY</h4>
                                <span className="mono">{(results.strategies.risky_strategy?.total_optimal_delta || 0).toFixed(1)}s DELTA</span>
                            </div>
                            <div className="strat-philosophy">Extends stints past cliff · Gambles on drys in light rain · Pits under SC for faster tires.</div>
                            <StrategyBar strategy={results.strategies.risky_strategy} totalLaps={results.strategies.risky_strategy?.stints_data?.reduce((a, b) => a + b.laps, 0) || 1} />
                            <div className="strat-sequence" style={{fontSize: '0.8rem', opacity: 0.6}}>{results.strategies.risky_strategy?.sequence}</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
      )}
    </div>
  );
}

export default App;
