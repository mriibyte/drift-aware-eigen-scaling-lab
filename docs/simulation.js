/* Shared numerical core. Browser global + CommonJS export for Node tests. */
(function (root, factory) {
  const simulation = factory();
  if (typeof module === 'object' && module.exports) module.exports = simulation;
  else root.EigenSimulation = simulation;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const DEFAULTS = Object.freeze({ steps: 260, curvature: 1, motion: 1, plainEta: 0.035, gain: 1.78, safety: 0.10, power: 7 });
  const range = n => Array.from({ length: n }, (_, index) => index);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const norm = a => Math.hypot(a[0], a[1]);
  const matVec = (matrix, vector) => [matrix[0][0] * vector[0] + matrix[0][1] * vector[1], matrix[1][0] * vector[0] + matrix[1][1] * vector[1]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const finite = Number.isFinite;

  function spectrum(t, steps, factor) {
    const phase = 2 * Math.PI * t / Math.max(steps - 1, 1);
    return { min: 0.65 + 0.18 * Math.sin(phase + 0.75), max: 8 + 13 * factor * (0.5 + 0.5 * Math.sin(phase - 0.85)) + 11 * factor * t / Math.max(steps - 1, 1) };
  }
  function rotation(t, steps) {
    const phase = 2 * Math.PI * t / Math.max(steps - 1, 1);
    return 0.18 + 0.70 * Math.sin(0.55 * phase) + 0.12 * Math.sin(phase);
  }
  function optimum(t, steps, motion) {
    const phase = 2 * Math.PI * t / Math.max(steps - 1, 1);
    return [1.35 * Math.cos(0.72 * phase * motion), 0.88 * Math.sin(0.72 * phase * motion + 0.55)];
  }
  function hessian(t, config) {
    const eigenvalues = spectrum(t, config.steps, config.curvature);
    const theta = rotation(t, config.steps), c = Math.cos(theta), s = Math.sin(theta);
    return [[c * c * eigenvalues.min + s * s * eigenvalues.max, c * s * (eigenvalues.min - eigenvalues.max)], [c * s * (eigenvalues.min - eigenvalues.max), s * s * eigenvalues.min + c * c * eigenvalues.max]];
  }
  function objective(x, t, config) {
    if (!x || !x.every(finite)) return NaN;
    const delta = sub(x, optimum(t, config.steps, config.motion));
    return 0.5 * dot(delta, matVec(hessian(t, config), delta));
  }

  function validate(config) {
    if (!Number.isInteger(config.steps) || config.steps < 10 || config.steps > 5000) throw new RangeError('Iterations must be an integer between 10 and 5000.');
    if (!Number.isInteger(config.power) || config.power < 1 || config.power > 100) throw new RangeError('Power iterations must be an integer between 1 and 100.');
    for (const key of ['curvature', 'motion', 'plainEta', 'gain', 'safety']) {
      if (!finite(config[key])) throw new RangeError(`${key} must be finite.`);
    }
    if (config.curvature < 0 || config.motion < 0 || config.safety < 0 || config.plainEta <= 0 || config.gain <= 0) throw new RangeError('Drift, motion and safety must be nonnegative; step and gain must be positive.');
  }

  function runMethod(method, config, start) {
    const n = config.steps, states = [start.slice()], gaps = [objective(start, 0, config)];
    const distances = [norm(sub(start, optimum(0, n, config.motion)))];
    const eta = Array(n).fill(NaN), rawLambda = Array(n).fill(NaN), trackedLambda = Array(n).fill(NaN), driftScore = Array(n).fill(NaN);
    const initialLambda = spectrum(0, n, config.curvature).max;
    let x = start.slice(), divergedAt = null;
    let vector = [0.83, 0.41], estimate = null, previousRaw = null;
    for (let t = 0; t < n; t++) {
      const H = hessian(t, config);
      let step;
      if (method === 'normal') step = config.plainEta;
      else if (method === 'frozen') step = config.gain / initialLambda;
      else if (method === 'drift') {
        for (let j = 0; j < config.power; j++) {
          const candidate = matVec(H, vector), length = norm(candidate);
          if (!length) break;
          vector = [candidate[0] / length, candidate[1] / length];
        }
        const raw = dot(vector, matVec(H, vector));
        rawLambda[t] = raw;
        let drift = 0;
        if (estimate === null) estimate = raw;
        else {
          drift = Math.abs(raw - previousRaw) / Math.max(Math.abs(previousRaw), 1e-12);
          const alpha = Math.min(0.82, 0.16 + 7 * drift);
          let filtered = (1 - alpha) * estimate + alpha * raw;
          if (raw > estimate) filtered = Math.max(filtered, raw);
          estimate = filtered;
        }
        driftScore[t] = drift;
        trackedLambda[t] = estimate;
        step = config.gain / ((1 + config.safety) * estimate);
        previousRaw = raw;
      } else throw new Error(`Unknown method: ${method}`);

      eta[t] = step;
      const gradient = matVec(H, sub(x, optimum(t, n, config.motion)));
      x = [x[0] - step * gradient[0], x[1] - step * gradient[1]];
      states.push(x.slice());
      // x_(t+1) is assessed against the next objective; the last snapshot holds.
      const at = Math.min(t + 1, n - 1);
      gaps.push(objective(x, at, config));
      distances.push(norm(sub(x, optimum(at, n, config.motion))));
      if (!x.every(finite) || norm(x) > 1e6) { divergedAt = t + 1; break; }
    }
    if (method !== 'drift') { rawLambda.fill(initialLambda); trackedLambda.fill(initialLambda); }
    return { method, states, gaps, distances, eta, rawLambda, trackedLambda, driftScore, divergedAt };
  }
  function simulate(config, start = [-2.35, 2.05]) {
    validate(config);
    if (!Array.isArray(start) || start.length !== 2 || !start.every(finite)) throw new RangeError('The start point must have two finite coordinates.');
    return { normal: runMethod('normal', config, start), frozen: runMethod('frozen', config, start), drift: runMethod('drift', config, start) };
  }

  // All paths and surface cells use the SAME f_0 and parameter units. Never
  // normalize each coordinate independently or silently remap divergent states.
  function landscape(config, limit = 3.5, samples = 33) {
    const axis = range(samples).map(index => -limit + 2 * limit * index / (samples - 1));
    const z = axis.map(y => axis.map(x => objective([x, y], 0, config)));
    return { x: axis, y: axis, z, max: Math.max(...z.flat()) };
  }
  function visibleTrajectory(result, config, view, limit = 3.5) {
    const x = [], y = [], z = [], customdata = [];
    result.states.forEach((point, t) => {
      const visible = point.every(finite) && point.every(value => Math.abs(value) <= limit);
      x.push(visible ? point[0] : null);
      y.push(visible ? point[1] : null);
      z.push(visible ? (view === 'time' ? t : objective(point, 0, config)) : null);
      customdata.push([t, result.gaps[t]]);
    });
    return { x, y, z, customdata, clipped: x.filter(value => value === null).length };
  }
  return { DEFAULTS, range, spectrum, rotation, optimum, hessian, objective, simulate, landscape, visibleTrajectory };
});
