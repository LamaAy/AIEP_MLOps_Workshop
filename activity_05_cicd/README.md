# Activity 05 — CI Quality Gate Challenge (GitHub Actions)

**Goal:** Protect the Fashion-MNIST FastAPI service from regressions by running automated checks on every pull request. This activity implements **Continuous Integration (CI)**, not automatic deployment.

## Learning objectives

- Explain triggers, jobs, steps, and runners in GitHub Actions.
- Run code-quality checks (Ruff) and automated API tests (pytest).
- Introduce a genuine application bug, observe CI fail, and fix the regression.
- Add a regression test and read workflow logs.

## Prerequisites

- A fork of this repository and permission to push to your fork.
- Python environment containing the Activity 03 dependencies.
- Run **all terminal commands from the repository root**.

## 1. Explore the CI pipeline

Open [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Its two jobs are:

1. `lint`: checks syntax and selected common code errors with Ruff.
2. `fastapi-tests`: installs dependencies, generates a **synthetic untrained checkpoint**, and runs the FastAPI regression tests.

The workflow runs on pushes to `main`, pull requests, or manual **Run workflow**. It does **not** run automatically on every push to a non-main branch unless that branch has an open pull request.

## 2. Run tests locally

Open a terminal in the repository root:

```bash
cd ~/AIEP_MLOps_Workshop
source .venv/bin/activate
python -m pip install -r activity_03_fastapi/requirements.txt
python -m pip install 'ruff>=0.9,<1'
ruff check activity_01_training activity_03_fastapi activity_05_cicd --select E4,E7,E9,F821,F822,F823,F831
python -m pytest -q activity_03_fastapi/test_api.py
```

The local API tests require `checkpoints/model.pth` from Activity 01. On GitHub Actions, the workflow creates an untrained checkpoint instead. **Do not run the CI checkpoint generator locally if you want to keep your trained Activity 01 checkpoint:** it writes to the same path.

## 3. Create a safe challenge branch

First ensure your working tree is clean with `git status` (commit or stash unrelated work before branching). Then:

```bash
git switch -c activity-05-quality-gate
```

Add a short comment to this README and commit it to start your exercise:

```bash
git add activity_05_cicd/README.md
git commit -m "Start Activity 05 CI challenge"
git push -u origin activity-05-quality-gate
```

On GitHub, open a **pull request** from the activity branch into `main`. Find the **MLOps CI** checks in the pull request, and inspect the logs under **Actions**. The checks should pass if the baseline application is correct.

## 4. Break the application (without changing the tests)

Open `activity_03_fastapi/main.py`. In the `/predict` response dictionary, temporarily replace:

```python
"class_id": class_id,
```

with:

```python
"class_id": -1,  # Deliberate bug for the exercise
```

Save the file and run locally:

```bash
python -m pytest -q activity_03_fastapi/test_api.py
```

The `test_predict` assertion should fail because valid class IDs are between 0 and 9. Commit and push **only to your challenge branch**:

```bash
git add activity_03_fastapi/main.py
git commit -m "Exercise: introduce API regression"
git push
```

Refresh the pull request and locate the failed `fastapi-tests` job. Read the assertion error.

## 5. Fix the bug and add a regression test

Restore the original line in `main.py`:

```python
"class_id": class_id,
```

Review the additional tests in `activity_03_fastapi/test_api.py`, including empty images, corrupt files, JPEG uploads, missing files, and oversized uploads.

**Your challenge:** Write one more pytest test for a meaningful API requirement, such as checking that `predicted_class` is one of the ten Fashion-MNIST classes. Confirm that it passes on correct code. To prove it detects regressions, temporarily break the related response, observe the failing test, and restore the correct code before committing.

Then:

```bash
python -m pytest -q activity_03_fastapi/test_api.py
git add activity_03_fastapi/main.py activity_03_fastapi/test_api.py
git commit -m "Fix API regression and add coverage"
git push
```

Your pull request should now show passing checks. Do not merge the intentionally broken version into `main`.

## 6. Reflect

1. Which events trigger the workflow?
2. What is the difference between the lint and API-test jobs?
3. Why is a synthetic checkpoint sufficient for API contract tests but not for model accuracy evaluation?
4. Which assertion detected the intentional regression?
5. What would you add to validate a real trained model before deployment?

## Expected output

- A pull request showing a failed CI run followed by a successful run.
- Working Ruff and pytest jobs.
- At least one additional regression test.
- An explanation of what the quality gate protects.

**Scope:** This CI workflow tests software behavior only. It neither retrains the CNN nor measures predictive accuracy; model quality needs a separate evaluation using a genuinely trained checkpoint.
