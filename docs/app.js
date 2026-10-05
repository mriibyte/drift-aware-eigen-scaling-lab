/* UI and Plotly rendering. Numerical code is kept in simulation.js. */
(() => {
  'use strict';
  const Sim = window.EigenSimulation;
  const COLORS = { normal: '#6b7c93', frozen: '#ed8354', drift: '#008f94', purple: '#8b61b8', ink: '#102b46' };
  const METHODS = [
    { key: 'normal', label: 'Normal GD', color: COLORS.normal, dash: 'dash', lane: 0 },
    { key: 'frozen', label: 'Frozen scale', color: COLORS.frozen, dash: 'dot', lane: 1 },
    { key: 'drift', label: 'Drift-aware', color: COLORS.drift, dash: 'solid', lane: 2 }
  ];
  const $ = id => document.getElementById(id);
  const finite = Number.isFinite;
  const safeLog = value => finite(value) ? Math.log10(Math.max(value, 1e-14)) : null;
  const lastFinite = values => values.findLast(finite);
  const sci = value => !finite(value) ? '—' : (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01) ? value.toExponential(2) : value.toFixed(4));
  const PLOT_IDS = ['trajectory', 'loss', 'spectrum', 'step', 'error', 'drift'];
  const PLOT_CONFIG = { responsive: true, displaylogo: false, scrollZoom: true, displayModeBar: true, toImageButtonOptions: { format: 'png', filename: 'eigen-scaling-lab', scale: 2 } };
  const AXIS = { gridcolor: '#dce6ed', zerolinecolor: '#b9cad5', linecolor: '#a7bac8', tickfont: { color: '#607389', size: 10 }, title: { font: { color: '#102b46', size: 12 } }, showbackground: true, backgroundcolor: '#f8fbfc' };
  let latest = null, timer = null, busy = false, pending = false, revision = 0;

  function status(message, updating = false) {
    $('status').textContent = message;
    $('status').classList.toggle('live', updating);
  }
  function readConfig() {
    return Object.fromEntries(Object.keys(Sim.DEFAULTS).map(id => [id, Number($(id).value)]));
  }
  function updateOutputs() {
    const config = readConfig();
    for (const [id, value] of Object.entries(config)) {
      let display = value;
      if (id === 'curvature' || id === 'motion') display = value.toFixed(2) + '×';
      if (id === 'plainEta') display = value.toFixed(3);
      if (id === 'gain') display = value.toFixed(2);
      if (id === 'safety') display = Math.round(value * 100) + '%';
      $(id + 'Out').textContent = display;
    }
  }
  function axis(title, overrides = {}) { return { ...AXIS, title: { text: title, font: AXIS.title.font }, ...overrides }; }
  function layout(id, axes, options = {}) {
    return {
      autosize: true, height: $(id).clientHeight, uirevision: options.revision || id,
      margin: { l: 8, r: 8, t: 18, b: 78 }, paper_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', color: '#102b46', size: 11 },
      legend: { orientation: 'h', x: 0, y: -0.12, xanchor: 'left', yanchor: 'top', font: { size: 11 }, itemclick: 'toggle', itemdoubleclick: 'toggleothers' },
      scene: {
        bgcolor: '#ffffff', dragmode: 'orbit', aspectmode: 'manual', aspectratio: options.aspect || { x: 1.6, y: 0.9, z: 1 },
        camera: { eye: { x: 1.55, y: 1.65, z: 1.1 } },
        xaxis: axes.x, yaxis: axes.y, zaxis: axes.z
      }
    };
  }
  function line(name, coordinates, color, dash, template) {
    const { x, y, z, customdata } = coordinates;
    return { type: 'scatter3d', mode: 'lines', name, x, y, z, customdata, connectgaps: false, line: { color, width: 5, dash: dash || 'solid' }, hovertemplate: template };
  }
  function plot(id, traces, figureLayout) { return window.Plotly.react($(id), traces, figureLayout, PLOT_CONFIG); }

  function trajectory(config, results) {
    const view = $('trajectoryView').value;
    const reference = Sim.landscape(config);
    const traces = [];
    if (view === 'landscape') {
      traces.push({ type: 'surface', name: 'Initial objective f₀', x: reference.x, y: reference.y, z: reference.z,
        showscale: false, opacity: 0.76, colorscale: [[0, '#eff8fa'], [0.25, '#bddde5'], [1, '#427291']],
        contours: { z: { show: true, usecolormap: true, project: { z: true } } },
        hovertemplate: 'Initial landscape<br>x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>f₀=%{z:.4g}<extra></extra>'
      });
    }
    const ts = Sim.range(config.steps + 1), centers = ts.map(t => Sim.optimum(Math.min(t, config.steps - 1), config.steps, config.motion));
    traces.push(line('Moving target (reference)', { x: centers.map(p => p[0]), y: centers.map(p => p[1]), z: centers.map((p, t) => view === 'time' ? t : Sim.objective(p, 0, config)), customdata: ts }, '#a2b0ba', 'dash', 'Moving target<br>iteration %{customdata}<br>x₁=%{x:.3f}<br>x₂=%{y:.3f}<extra></extra>'));
    const clipped = [];
    for (const meta of METHODS) {
      const coordinates = Sim.visibleTrajectory(results[meta.key], config, view);
      if (coordinates.clipped) clipped.push(meta.label);
      traces.push(line(meta.label, coordinates, meta.color, meta.dash,
        `${meta.label}<br>iteration %{customdata[0]}<br>x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>${view === 'time' ? 't' : 'f₀'}=%{z:.4g}<br>instantaneous gap=%{customdata[1]:.4g}<extra></extra>`));
      const final = coordinates.x.length - 1;
      if (coordinates.x[final] !== null) {
        traces.push({ type: 'scatter3d', mode: 'markers', name: meta.label + ' end', showlegend: false,
          x: [coordinates.x[final]], y: [coordinates.y[final]], z: [coordinates.z[final]],
          marker: { size: 5, color: meta.color }, hovertemplate: meta.label + ' final visible point<extra></extra>' });
      }
    }
    const initial = Sim.optimum(0, config.steps, config.motion);
    traces.push({ type: 'scatter3d', mode: 'markers', name: 'Initial optimum', x: [initial[0]], y: [initial[1]], z: [0], marker: { color: COLORS.ink, size: 7, symbol: 'diamond' }, hovertemplate: 'Initial optimum<extra></extra>' });
    $('trajectoryCaption').textContent = view === 'landscape'
      ? 'Surface and paths use the same initial objective f₀(x₁, x₂). The objective changes during the run; hover for instantaneous gaps.'
      : 'x₁ and x₂ are the original parameter coordinates; height is iteration. Both parameter axes have equal scale.';
    $('trajectoryNote').textContent = clipped.length
      ? `Outside ±3.5: ${clipped.join(', ')}. Those path segments are hidden, not rescaled. The comparison charts retain their full recorded gaps.`
      : 'Parameter axes share the same scale. Drag to orbit, scroll/pinch to zoom, and hover to inspect a point.';
    return plot('trajectory', traces, layout('trajectory', {
      x: axis('Parameter x₁', { range: [-3.5, 3.5] }), y: axis('Parameter x₂', { range: [-3.5, 3.5] }),
      z: axis(view === 'time' ? 'Iteration t' : 'Initial objective f₀(x)', { range: view === 'time' ? [0, config.steps] : [0, reference.max * 1.03] })
    }, { revision: 'trajectory-' + view, aspect: { x: 1, y: 1, z: 0.75 } }));
  }
  function laneChart(id, series, zTitle, steps, options = {}) {
    const traces = series.map((entry, index) => {
      const raw = entry.values.map(value => finite(value) ? value : null);
      const heights = raw.map(value => value === null ? null : (options.log ? safeLog(value) : value));
      return line(entry.name, { x: Sim.range(raw.length), y: Array(raw.length).fill(index), z: heights, customdata: raw }, entry.color, entry.dash,
        `${entry.name}<br>iteration %{x}<br>${options.log ? 'log₁₀ height=%{z:.3f}<br>' : ''}value=%{customdata:.5g}<extra></extra>`);
    });
    return plot(id, traces, layout(id, {
      x: axis('Iteration t', { range: [0, steps] }),
      y: axis('Series (categorical)', { tickmode: 'array', tickvals: Sim.range(series.length), ticktext: series.map(entry => entry.name), range: [-0.3, series.length - 0.7] }),
      z: axis(zTitle)
    }));
  }
  function seriesFor(results, field) { return METHODS.map(meta => ({ name: meta.label, values: results[meta.key][field], color: meta.color, dash: meta.dash })); }
  function stepChart(config, results, safe) {
    // A limit surface spans every method lane: unlike separated limit lines,
    // this gives an actual common height against which each step is compared.
    const ts = Sim.range(config.steps), lanes = [-0.3, 2.3];
    const traces = [{ type: 'surface', name: 'Stability ceiling', x: ts, y: lanes, z: [safe, safe], colorscale: [[0, '#d6e4ef'], [1, '#94b1c5']], opacity: 0.4, showscale: false,
      hovertemplate: 'Fixed-quadratic stability limit<br>iteration %{x}<br>2/λmax=%{z:.5f}<extra></extra>' }];
    for (const meta of METHODS) {
      traces.push(line(meta.label, { x: ts, y: Array(ts.length).fill(meta.lane), z: results[meta.key].eta.map(value => finite(value) ? value : null) }, meta.color, meta.dash,
        `${meta.label}<br>iteration %{x}<br>step η=%{z:.5f}<extra></extra>`));
    }
    return plot('step', traces, layout('step', {
      x: axis('Iteration t', { range: [0, config.steps] }),
      y: axis('Method', { tickmode: 'array', tickvals: [0, 1, 2], ticktext: METHODS.map(meta => meta.label), range: lanes }),
      z: axis('Step size η')
    }));
  }
  function driftChart(config, results) {
    const ts = Sim.range(config.steps);
    const peak = Math.max(...results.drift.trackedLambda.filter(finite));
    const peakDrift = Math.max(...results.drift.driftScore.filter(finite), 1e-12);
    const signals = [
      { name: 'Tracked curvature / peak', raw: results.drift.trackedLambda, divisor: peak, color: COLORS.drift },
      { name: 'Relative drift / peak', raw: results.drift.driftScore, divisor: peakDrift, color: COLORS.purple }
    ];
    const traces = signals.map((signal, lane) => line(signal.name, { x: ts, y: Array(ts.length).fill(lane), z: signal.raw.map(value => finite(value) ? value / signal.divisor : null), customdata: signal.raw }, signal.color, 'solid', `${signal.name}<br>iteration %{x}<br>normalized=%{z:.3f}<br>raw=%{customdata:.5g}<extra></extra>`));
    return plot('drift', traces, layout('drift', { x: axis('Iteration t', { range: [0, config.steps] }), y: axis('Signal', { tickmode: 'array', tickvals: [0, 1], ticktext: ['Curvature', 'Spectral change'], range: [-0.2, 1.2] }), z: axis('Each signal / its own peak', { range: [0, 1.05] }) }));
  }
  function summaries(config, results, trueMax) {
    for (const meta of METHODS.slice(0, 1).concat(METHODS.slice(2))) {
      const result = results[meta.key], prefix = meta.key === 'normal' ? 'normal' : 'drift';
      $(prefix + 'Gap').textContent = sci(lastFinite(result.gaps));
      $(prefix + 'Status').textContent = result.divergedAt === null ? 'Completed · unscaled fixed step' : `Stopped: divergent at t=${result.divergedAt}`;
      if (prefix === 'drift' && result.divergedAt === null) $(prefix + 'Status').textContent = 'Completed · adaptive eigen scaling';
    }
    $('frozenStatus').textContent = results.frozen.divergedAt === null ? 'Completed' : `Diverged t=${results.frozen.divergedAt}`;
    $('peakLambda').textContent = Math.max(...trueMax).toFixed(2);
  }
  async function renderAll() {
    const { config, results } = latest;
    const ts = Sim.range(config.steps), trueMax = ts.map(t => Sim.spectrum(t, config.steps, config.curvature).max);
    summaries(config, results, trueMax);
    // Render sequentially to avoid competing WebGL uploads on mobile devices.
    await trajectory(config, results);
    await laneChart('loss', seriesFor(results, 'gaps'), 'log₁₀ objective gap', config.steps, { log: true });
    await laneChart('spectrum', [
      { name: 'True λmax', values: trueMax, color: COLORS.ink },
      { name: 'True λmin', values: ts.map(t => Sim.spectrum(t, config.steps, config.curvature).min), color: '#739eb6' },
      { name: 'Raw power estimate', values: results.drift.rawLambda, color: COLORS.purple, dash: 'dot' },
      { name: 'Tracked λmax', values: results.drift.trackedLambda, color: COLORS.drift }
    ], 'Eigenvalue', config.steps);
    await stepChart(config, results, trueMax.map(value => 2 / value));
    await laneChart('error', seriesFor(results, 'distances'), 'log₁₀ distance', config.steps, { log: true });
    await driftChart(config, results);
  }
  function errorState(error) {
    status('Unable to render — see message below');
    $('plotError').hidden = false;
    $('plotError').textContent = 'The 3D charts require WebGL and Plotly.js. Check your network/browser graphics support and reload. ' + error.message;
  }
  async function flush() {
    timer = null;
    if (busy) return;
    pending = false; busy = true;
    status('Live · updating…', true);
    try {
      if (!window.Plotly) throw new Error('Plotly.js failed to load.');
      const config = readConfig();
      latest = { config, results: Sim.simulate(config) };
      await renderAll();
      revision++;
      document.documentElement.dataset.simulationRevision = String(revision);
      $('plotError').hidden = true;
      status('Live · up to date');
    } catch (error) { errorState(error); }
    finally {
      busy = false;
      // A newer input never gets lost: finish this frame, then render the most
      // recent settings. No overlapping Plotly.react calls or stale timers.
      if (pending) timer = window.setTimeout(flush, 80);
    }
  }
  function schedule(immediate = false) {
    updateOutputs(); pending = true; status('Live · updating…', true);
    if (!busy && timer === null) timer = window.setTimeout(flush, immediate ? 0 : 80);
  }
  $('run').addEventListener('click', () => schedule(true));
  $('reset').addEventListener('click', () => { for (const [id, value] of Object.entries(Sim.DEFAULTS)) $(id).value = value; schedule(true); });
  $('trajectoryView').addEventListener('change', () => schedule(true));
  for (const input of document.querySelectorAll('input[type=range]')) input.addEventListener('input', () => schedule());
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (!window.Plotly) return;
      for (const id of PLOT_IDS) {
        const element = $(id);
        if (element._fullLayout) window.Plotly.relayout(element, { height: element.clientHeight, autosize: true });
      }
    }, 120);
  });
  updateOutputs(); schedule(true);
})();
