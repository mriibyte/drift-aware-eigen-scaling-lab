# Contributing

Thanks for helping improve the lab.

## Before opening a change

1. Keep numerical claims tied to a reproducible configuration and seed.
2. Distinguish the 2-D quadratic tracker from the separate classification HVP-norm scripts.
3. Do not describe synthetic or exploratory results as clinical evidence.
4. Add or update a regression test when changing numerical behavior.

## Local checks

```bash
npm ci
npm test
npm run test:python
npx playwright install chromium
npm run test:browser
```

Keep generated PNGs, Playwright reports, virtual environments, and local credentials out of commits. See `.gitignore`.
