# Contributing to sympldiff

We welcome community contributions, bug reports, feature suggestions, and pull requests! `sympldiff` is developed to provide robust, mathematically sound, structure-preserving diffeomorphic operators for medical imaging, deep learning, and computational anatomy.

---

## Code of Conduct
Please be polite, constructive, and respectful in all interactions, issues, and discussions.

---

## How to Report Bugs
If you find a bug or mathematical inconsistency:
1. Check existing GitHub Issues to ensure it hasn't already been reported.
2. Open a new issue with a clear title and description.
3. Provide a minimal reproducible example (code snippet, inputs, expected output vs. actual output).
4. Specify your Python version, PyTorch version, OS, and hardware (CPU/GPU).

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sympldiff/sympldiff.git
   cd sympldiff
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run the test suite:**
   ```bash
   pytest tests/
   ```

---

## Pull Request Guidelines

1. **Keep PRs focused:** Submit small, self-contained PRs addressing a single feature or bug fix.
2. **Include unit tests:** Every new operator, layer, or boundary condition must include unit tests verifying:
   - Volume conservation ($\det(D\Phi) \equiv 1$ or prescribed density).
   - Invertibility ($\Phi^{-1}(\Phi(x)) \approx x$).
   - Boundary invariance (points inside the domain never escape).
   - Reversible autograd gradient equivalence.
3. **Ensure tests pass:** Run `pytest tests/` locally before opening the PR.
4. **Follow code style:** Keep docstrings complete, adhere to PEP 8, and use clear mathematical parameter names.

---

## Questions and Support
For questions regarding the mathematical formulation or API usage, feel free to open a Discussion on GitHub.
