# Activity 05 — CI with GitHub Actions

Automate the FastAPI tests from Activity 03 whenever code is pushed or a pull request is opened.

## Learning objectives
- Create a GitHub Actions workflow in `.github/workflows/ci.yml`.
- Run automated API tests in a fresh environment.
- Interpret passed and failed checks in GitHub Actions.

## 1. Push the activity

From the repository root:

```bash
git add .github/workflows/ci.yml activity_05_cicd README.md
git commit -m "Add Activity 05 - GitHub Actions CI"
git pull --rebase origin main
git push origin main
```

## 2. View results

Open your GitHub repository → **Actions** → **MLOps CI** → select the latest run.

Expected: all four Activity 03 API tests pass on the GitHub-hosted Ubuntu runner.

## 3. Student challenge

Create a branch, change the expected response code in one test, push it, and observe a failed check. Restore the test and push again. Avoid deliberately breaking `main`.

## What CI validates

The workflow generates a **synthetic, untrained CNN checkpoint** only to test loading, endpoint behavior, request validation, and response structure. It does **not** measure predictive accuracy or retrain the CNN. Your real Activity 01 checkpoint remains local and is not uploaded to GitHub.

## Questions

1. What triggers this workflow?
2. What is the difference between a failed test and a model with poor accuracy?
3. Why should model evaluation be a separate validation step?
