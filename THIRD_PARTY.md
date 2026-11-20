# Third-party software

- The bundled `assets/plotly.min.js` is Plotly.js 3.7.0, distributed under the MIT license. Its embedded notices are retained; the license is in `assets/LICENSE.plotly.txt`. Source: https://github.com/plotly/plotly.js.
- Pyodide 0.28.3 is loaded from its jsDelivr distribution at runtime. Pyodide is distributed under MPL-2.0 and includes CPython and other components under their respective licenses. Source and notices: https://github.com/pyodide/pyodide/tree/0.28.3.
- The runtime downloads NumPy 2.2.5, SciPy 1.14.1, NetworkX 3.4.2 and their Pyodide package dependencies. These projects retain their own copyright and license notices inside their packages. NumPy/SciPy/NetworkX use BSD licenses.
- The optional developer-test dependencies jsdom and the Pyodide npm package are recorded in `package.json`. Node/npm dependencies are not needed for deployment.

The desktop variant lists its installation requirements in `desktop/requirements.txt`.
