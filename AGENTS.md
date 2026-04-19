# AGENTS.md

## Repo Shape
- Single-package Django repo. `manage.py` uses `mysite.settings`.
- `mysite/` is the Django project scaffold; `Grader/` holds the unfinished app code; `Grader/machine_learning/` is the real ML/scraper entrypoint area.
- Root routing currently only exposes `admin/` in `mysite/urls.py`. `Grader/urls.py` exists but is empty and not included.

## Commands
- Use `python3` in this environment; `python` is not available.
- Install deps: `python3 -m pip install -r requirements.txt`
- Django startup order: `python3 manage.py migrate` then `python3 manage.py runserver`
- Cheapest Django verification after edits: `python3 manage.py check`
- Train model: `python3 Grader/machine_learning/train.py training_data/`
- Grade one image: `python3 Grader/machine_learning/grader.py path/to/card.jpg`
- Compare listings: `python3 Grader/machine_learning/compare.py "PSA 10 Charizard Base Set"`

## Verified Gotchas
- Trust executable config over README version claims: `requirements.txt` specifies `django>=4.2`, while `README.md` says Django 6.
- There is no committed lint, formatter, typecheck, pytest, pre-commit, or CI config. Do not invent a repo-standard command chain that is not present.
- No test files are committed (`tests.py` / `test_*.py` were not found). Prefer targeted smoke checks over assuming a real test suite exists.
- `Grader/views.py` is not currently runnable as wired: `Grader` and `rest_framework` are missing from `INSTALLED_APPS`, `settings.MEDIA_ROOT` is undefined, and the file imports `.ml.predict` even though the package on disk is `Grader/machine_learning/`.
- `Grader/machine_learning/predict.py` hardcodes `grader/machine_learning/checkpoints/psa_grader.keras` with lowercase `grader`; verify paths before relying on it outside this macOS workspace.
- README and docstrings mention `scraper_playwright.py`, but the scraper file in the repo is `Grader/machine_learning/scraper.py`.

## Mutable Artifacts
- Training outputs are written inside the source tree: `Grader/machine_learning/checkpoints/`, `Grader/machine_learning/training_data/`, `Grader/machine_learning/results.json`, `Grader/machine_learning/ebay_debug.html`, and root `db.sqlite3`.
- Treat those files as user data / generated artifacts. Do not delete or overwrite them unless the task requires it.
