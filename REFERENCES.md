# References and source notes

The following references are the bibliography used by the accompanying paper. They provide background or context; none should be read as a claim that the exact heuristic in this repository is novel or universally superior.

1. Jonathan Barzilai and Jonathan M. Borwein. “Two-point step size gradient methods.” *IMA Journal of Numerical Analysis*, 8(1), 141–148, 1988.
2. John Duchi, Elad Hazan, and Yoram Singer. “Adaptive subgradient methods for online learning and stochastic optimization.” *Journal of Machine Learning Research*, 12(61), 2121–2159, 2011. <https://jmlr.org/papers/v12/duchi11a.html>
3. Diederik P. Kingma and Jimmy Ba. “Adam: A method for stochastic optimization.” ICLR, 2015. <https://arxiv.org/abs/1412.6980>
4. Per-Gunnar Martinsson and Joel A. Tropp. “Randomized numerical linear algebra: Foundations and algorithms.” *Acta Numerica*, 29, 403–572, 2020. <https://doi.org/10.1017/S0962492920000021>
5. Aryan Mokhtari, Shahin Shahrampour, Ali Jadbabaie, and Alejandro Ribeiro. “Online optimization in dynamic environments: Improved regret rates for strongly convex problems.” CDC, 2016. <https://arxiv.org/abs/1603.04954>
6. Giorgio Pini and Giuseppe Gambolati. “Is a simple diagonal scaling the best preconditioner for conjugate gradients on supercomputers?” *Advances in Water Resources*, 13(3), 147–153, 1990.
7. B. T. Polyak. “Gradient methods for the minimisation of functionals.” *USSR Computational Mathematics and Mathematical Physics*, 3(4), 864–878, 1963.
8. Reza Takapoui and Hamid Javadi. “Preconditioning via diagonal scaling.” arXiv, 2016. <https://arxiv.org/abs/1610.03800>

## Third-party assets

- The browser visualizer vendors Plotly.js `2.35.2` in `docs/plotly-2.35.2.min.js`; it was obtained from the official Plotly CDN URL `https://cdn.plot.ly/plotly-2.35.2.min.js` so the deployed lab is not dependent on the CDN at runtime.
- JavaScript development tests use `@playwright/test`.
- Python experiments use NumPy, Matplotlib, pandas, and scikit-learn.

Consult each upstream project for its license and attribution requirements.
