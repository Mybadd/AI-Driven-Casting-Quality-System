import { useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

type CastingInput = {
  Alloy: string
  Pour_Temp: number
  Mold_Moisture: number
  Cooling_Time: number
  Riser: number
}
type Prediction = { probability?: number; threshold?: number; state?: string; decision?: string }
type Factor = { factor: string; importance: number; interpretation: string; review: string }
type Review = { statement?: string; factors?: Factor[] }
type AnalysisResult = {
  input?: CastingInput
  quality_predictions?: Record<string, Prediction>
  anomaly?: { status?: string; score?: number }
  engineering_reviews?: Record<string, Review>
  decision_note?: string
  [key: string]: unknown
}
type HistoryRecord = { id: number; created_at: string; input: CastingInput; result: AnalysisResult }
type Scenario = { scenario: CastingInput; predicted_probability: number; probability_change: number; interpretation: string }
type Tab = 'overview' | 'analysis' | 'results' | 'engineering' | 'scenarios' | 'history' | 'guide'

const initialInput: CastingInput = { Alloy: 'Al-Si', Pour_Temp: 710, Mold_Moisture: 3, Cooling_Time: 330, Riser: 1.25 }
const targets = [
  { key: 'Defect', title: 'Casting defects', description: 'Estimated likelihood of a casting defect.' },
  { key: 'Porosity', title: 'Porosity', description: 'Estimated likelihood of porosity-related issues.' },
  { key: 'Scrap', title: 'Scrap', description: 'Estimated likelihood of a scrap outcome.' },
]
function formatProbability(value: number | undefined) {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value <= 1 ? value * 100 : value).toFixed(1)}%`
    : 'Unavailable'
}

function App() {
  const [input, setInput] = useState<CastingInput>(initialInput)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [scenarioTarget, setScenarioTarget] = useState('Defect')
  const [scenarioLoading, setScenarioLoading] = useState(false)
  const [scenarioError, setScenarioError] = useState('')
  const [scenarioResult, setScenarioResult] = useState<{ baseline_probability?: number; scenarios_evaluated?: number; top_scenarios?: Scenario[]; decision_note?: string } | null>(null)
  const [history, setHistory] = useState<HistoryRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [selectedHistory, setSelectedHistory] = useState<HistoryRecord | null>(null)

  function updateInput<K extends keyof CastingInput>(key: K, value: CastingInput[K]) {
    setInput(current => ({ ...current, [key]: value }))
  }

  async function analyzeCasting(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setLoading(true); setError(''); setResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Analysis failed (${response.status}).`)
      setResult(data as AnalysisResult)
      setActiveTab('results')
    } catch (err) {
      setError(err instanceof Error ? `${err.message} Make sure the backend is running at ${API_BASE}.` : 'Unable to connect to the analysis service.')
    } finally { setLoading(false) }
  }

  async function exploreScenarios() {
    setScenarioLoading(true); setScenarioError(''); setScenarioResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/scenarios`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target: scenarioTarget, base_input: input,
          parameter_grid: { Pour_Temp: [700, 710, 720], Mold_Moisture: [2.8, 3.0, 3.2] }, top_n: 5,
        }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Scenario exploration failed (${response.status}).`)
      setScenarioResult(data)
    } catch (err) {
      setScenarioError(err instanceof Error ? err.message : 'Unable to explore scenarios.')
    } finally { setScenarioLoading(false) }
  }

  async function loadHistory() {
    setHistoryLoading(true); setHistoryError('')
    try {
      const response = await fetch(`${API_BASE}/api/history?limit=50`)
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Unable to load history (${response.status}).`)
      setHistory(Array.isArray(data.items) ? data.items as HistoryRecord[] : [])
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unable to load saved analyses.')
    } finally { setHistoryLoading(false) }
  }

  function navigate(tab: Tab) {
    setActiveTab(tab)
    if (tab === 'history') void loadHistory()
  }

  function renderInputForm() {
    return <>
      <div className="section-title-row"><div><h2>Process parameters</h2><p>Enter values within the training dataset ranges.</p></div><span className="step-label">INPUT · 01</span></div>
      <form className="input-card" onSubmit={analyzeCasting}>
        <div className="form-grid">
          <label className="field"><span className="field-title">Alloy type <span className="required">*</span></span><span className="field-help">Select the casting alloy</span>
            <select value={input.Alloy} onChange={e => updateInput('Alloy', e.target.value)} required>
              <option value="Al-Si">Al-Si — Aluminium silicon</option><option value="Al-Si-Mg">Al-Si-Mg — Aluminium silicon magnesium</option><option value="Al-Si-Cu">Al-Si-Cu — Aluminium silicon copper</option><option value="Al-Mg">Al-Mg — Aluminium magnesium</option>
            </select>
          </label>
          <label className="field"><span className="field-title">Pouring temperature *</span><span className="field-help">Temperature of molten metal</span><span className="input-with-unit"><input type="number" min="680" max="740" step="1" value={input.Pour_Temp} onChange={e => updateInput('Pour_Temp', Number(e.target.value))} required /><span>°C</span></span><span className="range-hint">Dataset range: 680–740 °C</span></label>
          <label className="field"><span className="field-title">Mold moisture *</span><span className="field-help">Moisture parameter for the mold</span><span className="input-with-unit"><input type="number" min="2.2" max="4" step="0.1" value={input.Mold_Moisture} onChange={e => updateInput('Mold_Moisture', Number(e.target.value))} required /><span>Index</span></span><span className="range-hint">Dataset range: 2.2–4.0</span></label>
          <label className="field"><span className="field-title">Cooling time *</span><span className="field-help">Cooling duration used in the process</span><span className="input-with-unit"><input type="number" min="250" max="450" step="1" value={input.Cooling_Time} onChange={e => updateInput('Cooling_Time', Number(e.target.value))} required /><span>Time units</span></span><span className="range-hint">Dataset range: 250–450</span></label>
          <label className="field"><span className="field-title">Riser parameter *</span><span className="field-help">Riser setting for the casting</span><span className="input-with-unit"><input type="number" min="1.05" max="1.55" step="0.01" value={input.Riser} onChange={e => updateInput('Riser', Number(e.target.value))} required /><span>Index</span></span><span className="range-hint">Dataset range: 1.05–1.55</span></label>
        </div>
        <div className="form-bottom"><p><span>ⓘ</span> Predictions support review; they do not confirm a defect or replace inspection.</p><button className="primary-button" type="submit" disabled={loading}>{loading ? 'Analyzing…' : 'Analyze casting'} <span>→</span></button></div>
      </form>
      {error && <div className="message error-message" role="alert"><strong>Analysis could not be completed</strong><p>{error}</p></div>}
    </>
  }

  function renderResults(data: AnalysisResult) {
    const anomaly = data.anomaly
    return <section className="results-section">
      {anomaly && <div className="anomaly-summary"><div><span className="anomaly-label">PROCESS PATTERN CHECK</span><h3>{anomaly.status === 'Normal' ? 'No unusual pattern flagged' : 'Unusual pattern flagged for review'}</h3><p>An unusual pattern does not automatically mean a defect.</p></div><span className={`anomaly-pill ${anomaly.status === 'Normal' ? 'normal' : 'flagged'}`}>{anomaly.status ?? 'Not available'}</span></div>}
      <div className="section-title-row"><div><h2>Quality estimates</h2><p>Model estimates for the submitted process conditions.</p></div><span className="step-label">RESULTS · 02</span></div>
      <div className="result-grid">{targets.map(target => {
        const prediction = data.quality_predictions?.[target.key]
        const probability = prediction?.probability
        const percent = typeof probability === 'number' ? Math.max(0, Math.min(100, probability <= 1 ? probability * 100 : probability)) : 0
        return <article className="result-card" key={target.key}><div className="result-card-top"><span className="result-symbol">◈</span><span className="result-label">MODEL ESTIMATE</span></div><h3>{target.title}</h3><p>{target.description}</p><div className="result-number">{formatProbability(probability)}</div><div className="decision-state">{prediction?.state ?? 'Review needed'}</div>{prediction?.decision && <p className="decision-description">{prediction.decision}</p>}<div className="result-bar"><span style={{ width: `${percent}%` }} /></div><div className="result-foot">Estimated positive-class probability</div></article>
      })}</div>
      <div className="message info-message"><strong>Interpret results carefully</strong><p>These model-based estimates are not guarantees of a defect or real-world frequency. Review alongside inspection, process records, and engineering knowledge.</p></div>
      <details className="raw-details"><summary>View technical response details</summary><pre>{JSON.stringify(data, null, 2)}</pre></details>
    </section>
  }

  function renderEngineering(data: AnalysisResult) {
    return <section className="engineering-panel"><div className="section-title-row"><div><h2>Engineering review guide</h2><p>Model-identified factors to investigate — not proven causes.</p></div></div>
      {targets.map(target => {
        const review = data.engineering_reviews?.[target.key]
        if (!review?.factors?.length) return null
        return <article className="engineering-card" key={target.key}><h3>{target.title}</h3>{review.statement && <p className="engineering-statement">{review.statement}</p>}
          {review.factors.map(factor => <div className="factor-row" key={factor.factor}><div className="factor-heading"><strong>{factor.factor}</strong><span>Importance {(factor.importance * 100).toFixed(1)}%</span></div><p>{factor.interpretation}</p><div className="review-action"><strong>Suggested review:</strong> {factor.review}</div></div>)}
        </article>
      })}
      <div className="message info-message"><strong>Important</strong><p>Feature importance describes model reliance, not causal effect or the direction a parameter should be changed.</p></div>
    </section>
  }

  function renderScenarios() {
    return <section className="scenario-panel"><div className="section-title-row"><div><h2>What-if scenario explorer</h2><p>Compare model estimates for alternative process settings.</p></div><span className="step-label">EXPLORE · 03</span></div>
      {!result && <div className="message info-message"><strong>Run an analysis first</strong><p>Current input values are used as the comparison baseline.</p><button className="primary-button" type="button" onClick={() => navigate('analysis')}>Enter process values →</button></div>}
      <div className="scenario-controls"><label className="field"><span className="field-title">Quality outcome to explore</span><span className="field-help">Choose an outcome</span><select value={scenarioTarget} onChange={e => setScenarioTarget(e.target.value)}><option value="Defect">Casting defects</option><option value="Porosity">Porosity</option><option value="Scrap">Scrap</option></select></label><button className="primary-button" type="button" onClick={() => void exploreScenarios()} disabled={scenarioLoading}>{scenarioLoading ? 'Comparing scenarios…' : 'Compare scenarios →'}</button></div>
      <p className="scenario-disclaimer">This compares temperatures of 700, 710, and 720 °C with mold moisture settings of 2.8, 3.0, and 3.2. These are model-based comparisons, not guaranteed optimal settings. Check actual process requirements before making changes.</p>
      {scenarioError && <div className="message error-message" role="alert"><strong>Scenario comparison failed</strong><p>{scenarioError}</p></div>}
      {scenarioResult && <div className="scenario-results"><div className="scenario-summary"><span>Current model estimate</span><strong>{formatProbability(scenarioResult.baseline_probability)}</strong><span>{scenarioResult.scenarios_evaluated ?? 0} combinations evaluated</span></div><h3>Best-ranked scenarios for this model</h3><div className="scenario-table-wrap"><table className="scenario-table"><thead><tr><th>Pour temp.</th><th>Mold moisture</th><th>Estimate</th><th>Change vs current</th><th>Model comparison</th></tr></thead><tbody>{(scenarioResult.top_scenarios ?? []).map((s, i) => <tr key={`${s.scenario.Pour_Temp}-${s.scenario.Mold_Moisture}-${i}`}><td>{s.scenario.Pour_Temp} °C</td><td>{s.scenario.Mold_Moisture.toFixed(1)}</td><td>{formatProbability(s.predicted_probability)}</td><td>{s.probability_change > 0 ? '+' : ''}{(s.probability_change * 100).toFixed(2)} percentage points</td><td>{s.interpretation}</td></tr>)}</tbody></table></div>{scenarioResult.decision_note && <p className="scenario-disclaimer">{scenarioResult.decision_note}</p>}</div>}
    </section>
  }

  function renderHistory() {
    return <section className="history-page"><div className="section-title-row"><div><h2>Analysis history</h2><p>Previously submitted analyses stored by the backend.</p></div><button className="secondary-button" type="button" onClick={() => void loadHistory()} disabled={historyLoading}>{historyLoading ? 'Refreshing…' : 'Refresh history'}</button></div>
      {historyError && <div className="message error-message" role="alert"><strong>Unable to load history</strong><p>{historyError}</p></div>}
      {historyLoading && <p>Loading saved analyses…</p>}
      {!historyLoading && !historyError && history.length === 0 && <div className="message info-message"><strong>No saved analyses yet</strong><p>Run an analysis; completed analyses are saved by the backend.</p></div>}
      {history.length > 0 && <div className="history-list">{history.map(record => <button className={`history-item ${selectedHistory?.id === record.id ? 'selected' : ''}`} type="button" key={record.id} onClick={() => setSelectedHistory(record)}><span className="history-item-title">Analysis #{record.id} · {record.input.Alloy}</span><span className="history-item-meta">{new Date(record.created_at).toLocaleString()}</span><span className="history-item-meta">{record.input.Pour_Temp} °C · Moisture {record.input.Mold_Moisture} · Cooling {record.input.Cooling_Time} · Riser {record.input.Riser}</span></button>)}</div>}
      {selectedHistory && <div className="history-detail"><div className="section-title-row"><div><h2>Saved analysis #{selectedHistory.id}</h2><p>Saved record details.</p></div><button className="secondary-button" type="button" onClick={() => { setInput(selectedHistory.input); setResult(selectedHistory.result); navigate('results') }}>Open results</button></div><pre>{JSON.stringify(selectedHistory.result, null, 2)}</pre></div>}
    </section>
  }

  const pageTitle: Record<Tab, string> = { overview: 'Casting quality', analysis: 'New analysis', results: 'Quality results', engineering: 'Engineering review', scenarios: 'What-if explorer', history: 'Analysis history', guide: 'Help & methodology' }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">CQ</div><div><strong>CastingIQ</strong><span>Quality Intelligence</span></div></div>
      <div className="sidebar-label">WORKSPACE</div>
      <button className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => navigate('overview')} type="button"><span>▦</span> Overview</button>
      <button className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => navigate('analysis')} type="button"><span>＋</span> New analysis</button>
      <button className={`nav-item ${activeTab === 'results' ? 'active' : ''}`} onClick={() => navigate('results')} type="button"><span>◉</span> Quality results</button>
      <button className={`nav-item ${activeTab === 'engineering' ? 'active' : ''}`} onClick={() => navigate('engineering')} type="button"><span>⌕</span> Engineering review</button>
      <button className={`nav-item ${activeTab === 'scenarios' ? 'active' : ''}`} onClick={() => navigate('scenarios')} type="button"><span>⇄</span> What-if explorer</button>
      <button className={`nav-item ${activeTab === 'history' ? 'active' : ''}`} onClick={() => navigate('history')} type="button"><span>◷</span> Analysis history</button>
      <button className={`nav-item ${activeTab === 'guide' ? 'active' : ''}`} onClick={() => navigate('guide')} type="button"><span>ⓘ</span> Help & methodology</button>
      <div className="sidebar-note"><div className="note-icon">✦</div><strong>Engineering support</strong><p>Explore model predictions and possible contributing factors. Always use engineering judgement.</p></div>
      <div className="sidebar-footer"><span className="status-dot" /> Student project prototype</div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> {pageTitle[activeTab]}</div><div className="topbar-right"><span className="topbar-chip">ML-powered analysis</span><div className="avatar">CQ</div></div></header>
      {activeTab === 'overview' && <>
        <section className="page-heading"><div><div className="eyebrow">FOUNDRY ENGINEERING • DECISION SUPPORT</div><h1>Casting quality overview</h1><p>Understand potential quality concerns before making an engineering decision.</p></div><div className="heading-badge"><span>●</span> Analysis workspace</div></section>
        <section className="welcome-banner"><div className="welcome-copy"><div className="banner-kicker">WELCOME TO CASTINGIQ</div><h2>Turn process data into<br />clearer quality insights.</h2><p>Enter casting conditions to explore predicted risks, review potential concerns, and compare possible process scenarios.</p><div className="banner-foot"><span>✓</span> Built for learning and engineering review</div><button className="primary-button" type="button" onClick={() => navigate('analysis')}>Start a new analysis →</button></div><div className="banner-illustration" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="cast-block"><div className="cast-top" /><div className="cast-face" /><div className="cast-side" /></div><div className="float-tag tag-one">AI <span>Analysis</span></div><div className="float-tag tag-two">↗ <span>Insights</span></div></div></section>
        <div className="section-title-row"><div><h2>Workspace at a glance</h2><p>Use the navigation to move between the system's screens.</p></div></div>
        <div className="guide-grid"><button className="guide-card" type="button" onClick={() => navigate('analysis')}><span>01</span><h3>New analysis</h3><p>Enter alloy and process conditions.</p></button><button className="guide-card" type="button" onClick={() => navigate('results')}><span>02</span><h3>Quality results</h3><p>{result ? 'View the latest model estimates.' : 'Run an analysis to populate results.'}</p></button><button className="guide-card" type="button" onClick={() => navigate('engineering')}><span>03</span><h3>Engineering review</h3><p>Review factors and suggested checks.</p></button><button className="guide-card" type="button" onClick={() => navigate('history')}><span>04</span><h3>Analysis history</h3><p>Browse saved database records.</p></button></div>
      </>}
      {activeTab === 'analysis' && <><section className="page-heading"><div><div className="eyebrow">INPUT WORKSPACE</div><h1>New casting analysis</h1><p>Submit process parameters to generate model-based estimates.</p></div></section>{renderInputForm()}</>}
      {activeTab === 'results' && <><section className="page-heading"><div><div className="eyebrow">MODEL OUTPUT</div><h1>Quality results</h1><p>Review estimates and anomaly signals from the latest analysis.</p></div></section>{result ? renderResults(result) : <div className="message info-message"><strong>No results yet</strong><p>Submit process values first.</p><button className="primary-button" type="button" onClick={() => navigate('analysis')}>Start analysis →</button></div>}</>}
      {activeTab === 'engineering' && <><section className="page-heading"><div><div className="eyebrow">INTERPRETATION SUPPORT</div><h1>Engineering review</h1><p>Use model explanations to guide investigation, not to establish causality.</p></div></section>{result ? renderEngineering(result) : <div className="message info-message"><strong>No review available yet</strong><p>Run an analysis to generate review guidance.</p><button className="primary-button" type="button" onClick={() => navigate('analysis')}>Start analysis →</button></div>}</>}
      {activeTab === 'scenarios' && <><section className="page-heading"><div><div className="eyebrow">CONSTRAINED COMPARISON</div><h1>What-if explorer</h1><p>Compare outputs for alternative process settings.</p></div></section>{renderScenarios()}</>}
      {activeTab === 'history' && <><section className="page-heading"><div><div className="eyebrow">PERSISTENT RECORDS</div><h1>Analysis history</h1><p>Browse analyses saved by the backend database.</p></div></section>{renderHistory()}</>}
      {activeTab === 'guide' && <section className="guide-page"><div className="eyebrow">QUICK GUIDE</div><h1>Help & methodology</h1><p className="guide-intro">CastingIQ is a student-project decision-support prototype for exploring casting quality estimates.</p><div className="guide-grid"><article className="guide-card"><span>01</span><h3>Enter process conditions</h3><p>Choose alloy and enter values within the displayed dataset ranges.</p></article><article className="guide-card"><span>02</span><h3>Run the analysis</h3><p>Values pass through trained ML pipelines using the backend API.</p></article><article className="guide-card"><span>03</span><h3>Review estimates</h3><p>Probabilities are indicators for investigation, not final verdicts.</p></article><article className="guide-card"><span>04</span><h3>Use engineering judgement</h3><p>Compare outputs with process records and physical inspection.</p></article></div><div className="message info-message"><strong>Important limitations</strong><p>Predictions are limited by training data. Feature importance is not proof of causality, scenario comparisons are not guaranteed optimal settings, and this tool must not be used for automatic rejection.</p></div></section>}
      <footer className="page-footer"><span>CASTINGIQ <span className="footer-dot">·</span> Academic engineering prototype</span><span>AI-assisted decision support, with human review</span></footer>
    </main>
  </div>
}

export default App
