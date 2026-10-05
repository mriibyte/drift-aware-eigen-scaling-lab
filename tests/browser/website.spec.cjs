const { test, expect } = require('@playwright/test');
const Sim = require('../../docs/simulation.js');

async function ready(page) {
  await expect(page.locator('#status')).toHaveText('Live · up to date');
  await expect(page.locator('#plotError')).toBeHidden();
}
async function revision(page) {
  return Number(await page.locator('html').getAttribute('data-simulation-revision'));
}
async function input(page, id, value) {
  await page.locator('#' + id).evaluate((element, newValue) => {
    element.value = String(newValue);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
}

test('aligned 3D charts, readable mathematics, live updates, camera persistence, and mobile layout', async ({ page }, testInfo) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await ready(page);
  await expect(page.locator('math')).toHaveCount(3);
  await expect(page.locator('.plot .gl-container')).toHaveCount(6);

  const checks = await page.evaluate(() => {
    const trajectory = document.getElementById('trajectory');
    return {
      aspect: trajectory._fullLayout.scene.aspectratio,
      xRange: trajectory._fullLayout.scene.xaxis.range,
      yRange: trajectory._fullLayout.scene.yaxis.range,
      axisTitles: [trajectory._fullLayout.scene.xaxis.title.text, trajectory._fullLayout.scene.yaxis.title.text],
      heights: ['loss','spectrum','step','error','drift'].map(id => {
        const plot = document.getElementById(id);
        return [plot.clientHeight, plot._fullLayout.height];
      }),
      colored: ['loss','spectrum','step','error','drift'].every(id => document.getElementById(id).data.filter(trace => trace.type === 'scatter3d').every(trace => typeof trace.line.color === 'string')),
      overflow: document.documentElement.scrollWidth > window.innerWidth
    };
  });
  expect(checks.aspect.x).toEqual(checks.aspect.y);
  expect(checks.xRange).toEqual(checks.yRange);
  expect(checks.axisTitles).toEqual(['Parameter x₁','Parameter x₂']);
  expect(checks.heights.every(([actual, plot]) => actual === plot)).toBeTruthy();
  expect(checks.colored).toBeTruthy();
  expect(checks.overflow).toBeFalsy();
  await page.screenshot({ path: testInfo.outputPath('desktop.png'), fullPage: true });

  // Exercise actual pointer rotation, then preserve that angle through reruns.
  await page.locator('#trajectory').scrollIntoViewIfNeeded();
  const initialCamera = await page.locator('#trajectory').evaluate(element => JSON.stringify(element._fullLayout.scene.camera));
  const plotBox = await page.locator('#trajectory').boundingBox();
  await page.mouse.move(plotBox.x + plotBox.width / 2, plotBox.y + plotBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(plotBox.x + plotBox.width / 2 + 90, plotBox.y + plotBox.height / 2 + 35, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const camera = await page.locator('#trajectory').evaluate(element => JSON.stringify(element._fullLayout.scene.camera));
  expect(camera).not.toBe(initialCamera);
  const initialGap = await page.locator('#normalGap').textContent();
  let previous = await revision(page);
  await input(page, 'plainEta', 0.008);
  await expect.poll(() => revision(page)).toBeGreaterThan(previous);
  await ready(page);
  expect(await page.locator('#normalGap').textContent()).not.toBe(initialGap);
  expect(await page.locator('#trajectory').evaluate(element => JSON.stringify(element._fullLayout.scene.camera))).toBe(camera);

  // Inputs continue during rendering. The last requested value must win and
  // there must be updates before the end of the input stream, not only on pause.
  previous = await revision(page);
  const updateCount = await page.evaluate(async () => {
    const original = Number(document.documentElement.dataset.simulationRevision);
    const input = document.getElementById('curvature');
    const end = performance.now() + 6000;
    let value = 0.5;
    while (performance.now() < end) {
      value = Math.min(1.8, value + 0.05);
      input.value = String(value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 55));
    }
    input.value = '1.8'; input.dispatchEvent(new Event('input', { bubbles: true }));
    return Number(document.documentElement.dataset.simulationRevision) - original;
  });
  expect(updateCount).toBeGreaterThan(0);
  await expect.poll(() => revision(page)).toBeGreaterThan(previous);
  const expectedPeakEnd = Sim.spectrum(Sim.DEFAULTS.steps - 1, Sim.DEFAULTS.steps, 1.8).max;
  await expect.poll(() => page.locator('#spectrum').evaluate(element => element.data[0].z.at(-1))).toBeCloseTo(expectedPeakEnd);
  await ready(page);
  await expect(page.locator('#curvatureOut')).toHaveText('1.80×');

  previous = await revision(page);
  await page.locator('#trajectoryView').selectOption('time');
  await expect.poll(() => revision(page)).toBeGreaterThan(previous);
  await ready(page);
  expect(await page.locator('#trajectory').evaluate(element => element.data.some(trace => trace.type === 'surface')))).toBeFalsy();
  expect(await page.locator('#trajectory').evaluate(element => element._fullLayout.scene.zaxis.title.text)).toBe('Iteration t');

  previous = await revision(page);
  await page.locator('#reset').click();
  await expect.poll(() => revision(page)).toBeGreaterThan(previous);
  await ready(page);
  await expect(page.locator('#plainEta')).toHaveValue('0.035');
  await expect(page.locator('#curvature')).toHaveValue('1');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(600);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  const mobilePlots = await page.locator('.plot').evaluateAll(elements => elements.map(element => [element.clientHeight, element._fullLayout.height]));
  expect(mobilePlots.every(([actual, plot]) => actual === plot)).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath('mobile.png'), fullPage: true });
  await page.locator('.theory').screenshot({ path: testInfo.outputPath('mobile-equations.png') });
  expect(errors).toEqual([]);
});

test('failure to load Plotly shows an explicit error rather than claiming success', async ({ page }) => {
  await page.route('https://cdn.plot.ly/**', route => route.abort());
  await page.goto('/');
  await expect(page.locator('#plotError')).toBeVisible();
  await expect(page.locator('#plotError')).toContainText('Plotly.js failed to load');
  await expect(page.locator('#status')).not.toHaveText('Live · up to date');
});
