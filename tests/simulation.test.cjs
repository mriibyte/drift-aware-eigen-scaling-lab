const test = require('node:test');
const assert = require('node:assert/strict');
const Sim = require('../docs/simulation.js');

const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test('Hessian has the promised positive spectrum and moving optimum has zero gap', () => {
  const config = { ...Sim.DEFAULTS };
  for (const t of [0, 1, 60, 150, 259]) {
    const H = Sim.hessian(t, config), spectrum = Sim.spectrum(t, config.steps, config.curvature);
    close(H[0][1], H[1][0]);
    close(H[0][0] + H[1][1], spectrum.min + spectrum.max);
    close(H[0][0] * H[1][1] - H[0][1] ** 2, spectrum.min * spectrum.max);
    assert.ok(spectrum.min > 0 && spectrum.max > spectrum.min);
    close(Sim.objective(Sim.optimum(t, config.steps, config.motion), t, config), 0);
  }
});

test('default numerical run retains the original comparison', () => {
  const results = Sim.simulate(Sim.DEFAULTS);
  assert.equal(results.normal.divergedAt, null);
  assert.equal(results.drift.divergedAt, null);
  assert.equal(results.frozen.divergedAt, 42);
  close(results.normal.gaps.at(-1), 0.06310092056731548);
  close(results.drift.gaps.at(-1), 0.03551662146367989);
  assert.equal(results.drift.states.length, Sim.DEFAULTS.steps + 1);
  assert.equal(results.drift.driftScore[0], 0);
});

test('slider changes affect numerical results and unsafe normal GD is reported', () => {
  const config = { ...Sim.DEFAULTS, curvature: 1.8, plainEta: 0.09 };
  const results = Sim.simulate(config);
  assert.notEqual(results.normal.divergedAt, null);
  assert.equal(results.drift.divergedAt, null);
  assert.notEqual(results.drift.gaps.at(-1), Sim.simulate(Sim.DEFAULTS).drift.gaps.at(-1));
});

test('minimum and maximum UI configurations produce bounded arrays with no infinities', () => {
  for (const config of [
    { steps:80, curvature:0.25, motion:0, plainEta:0.005, gain:0.5, safety:0, power:1 },
    { steps:500, curvature:1.8, motion:1.8, plainEta:0.09, gain:1.95, safety:0.35, power:12 }
  ]) {
    const results = Sim.simulate(config);
    for (const result of Object.values(results)) {
      assert.ok(result.states.length <= config.steps + 1);
      assert.equal(result.eta.length, config.steps);
      assert.ok(result.gaps.every(Number.isFinite));
    }
  }
});

test('reference surface and trajectory heights share the exact same coordinates/objective', () => {
  const config = { ...Sim.DEFAULTS }, result = Sim.simulate(config).drift;
  const surface = Sim.landscape(config);
  for (const [row, column] of [[0,0],[8,19],[32,32]]) {
    close(surface.z[row][column], Sim.objective([surface.x[column],surface.y[row]],0,config));
  }
  const coordinates = Sim.visibleTrajectory(result, config, 'landscape');
  for (const t of [0,1,25,260]) {
    close(coordinates.x[t], result.states[t][0]);
    close(coordinates.y[t], result.states[t][1]);
    close(coordinates.z[t], Sim.objective(result.states[t],0,config));
    assert.equal(coordinates.customdata[t][0], t);
    assert.equal(coordinates.customdata[t][1], result.gaps[t]);
  }
});

test('off-screen divergence is clipped without rescaling paths or renumbering time', () => {
  const result = { states: [[1,2],[9,10],[0,0]], gaps: [1,2,3] };
  const visible = Sim.visibleTrajectory(result, Sim.DEFAULTS, 'time');
  assert.deepEqual(visible.x, [1,null,0]);
  assert.deepEqual(visible.y, [2,null,0]);
  assert.deepEqual(visible.z, [0,null,2]);
  assert.equal(visible.clipped, 1);
});

test('invalid parameter values are rejected', () => {
  for (const overrides of [{steps:0},{power:0},{gain:NaN},{curvature:-1},{plainEta:0},{motion:Infinity}]) {
    assert.throws(() => Sim.simulate({...Sim.DEFAULTS,...overrides}), RangeError);
  }
});
