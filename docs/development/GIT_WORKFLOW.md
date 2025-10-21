# Git Workflow for SRE Analytics

**Effective Date:** 2025-10-12
**Status:** Active

---

## Overview

This project follows a **Feature Branch Workflow** with Pull Requests for all changes to the main branch.

---

## Branch Structure

### Main Branches

- **`main`** - Production-ready code, protected branch
  - All changes must come via Pull Requests
  - Requires review before merging
  - Always deployable

### Feature Branches

- **`feature/*`** - New features and enhancements
  - Format: `feature/short-description`
  - Examples: `feature/api-development`, `feature/ml-anomaly-detection`

- **`bugfix/*`** - Bug fixes
  - Format: `bugfix/issue-description`
  - Example: `bugfix/rate-limit-calculation`

- **`docs/*`** - Documentation updates
  - Format: `docs/what-changed`
  - Example: `docs/api-usage-guide`

- **`test/*`** - Test additions/improvements
  - Format: `test/what-testing`
  - Example: `test/api-authentication`

---

## Workflow Steps

### 1. Starting New Work

```bash
# Always start from latest main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-new-feature

# Or for bug fix
git checkout -b bugfix/fix-rate-limiting
```

### 2. Making Changes

```bash
# Make your changes, then stage and commit
git add <files>
git commit -m "feat: Add new feature description"

# Push to remote feature branch
git push origin feature/my-new-feature
```

### 3. Creating Pull Request

```bash
# After pushing, create PR on GitHub
# Visit: https://github.com/srk-sh1vkumar/sre-analytics/pulls

# Or use GitHub CLI
gh pr create --title "Add My New Feature" \
             --body "Description of changes" \
             --base main \
             --head feature/my-new-feature
```

### 4. Code Review & Merge

1. Request review from team members
2. Address review comments
3. Once approved, merge PR
4. Delete feature branch after merge

```bash
# After PR is merged, update local main
git checkout main
git pull origin main

# Delete local feature branch
git branch -d feature/my-new-feature
```

---

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **test**: Test additions/changes
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **chore**: Build/tooling changes

### Examples

```bash
# Feature
git commit -m "feat(api): Add authentication middleware"

# Bug fix
git commit -m "fix(ml): Correct anomaly threshold calculation"

# Documentation
git commit -m "docs: Update API usage examples"

# Test
git commit -m "test(api): Add rate limiting tests"

# Refactor
git commit -m "refactor: Extract metrics calculation to separate module"
```

---

## Retrospectively Created Branches

The following feature branches were created from existing commits:

### ✅ Pushed to Remote

1. **feature/testing-framework**
   - Commit: 9c51a43
   - Description: Comprehensive testing framework with 34 tests
   - Status: ✅ Pushed

2. **feature/appdynamics-integration**
   - Commit: 4ad1dff
   - Description: AppDynamics SLO integration with health scoring
   - Status: ✅ Pushed
   - PR URL: https://github.com/srk-sh1vkumar/sre-analytics/pull/new/feature/appdynamics-integration

3. **feature/prometheus-integration**
   - Commit: 531397c
   - Description: Prometheus SLO integration with PromQL support
   - Status: ✅ Pushed
   - PR URL: https://github.com/srk-sh1vkumar/sre-analytics/pull/new/feature/prometheus-integration

4. **feature/ml-anomaly-detection**
   - Commit: 806ba8a
   - Description: ML-based anomaly detection with 5 methods
   - Status: ✅ Pushed
   - PR URL: https://github.com/srk-sh1vkumar/sre-analytics/pull/new/feature/ml-anomaly-detection

5. **feature/api-development**
   - Commit: 85a6df9 (+ cleanup commit 3bc1982)
   - Description: RESTful API with FastAPI, auth, and rate limiting
   - Status: ✅ Pushed
   - PR URL: https://github.com/srk-sh1vkumar/sre-analytics/pull/new/feature/api-development

6. **feature/splunk-integration**
   - Commit: d712e9c
   - Description: Splunk integration for log aggregation
   - Status: ✅ Pushed
   - PR URL: https://github.com/srk-sh1vkumar/sre-analytics/pull/new/feature/splunk-integration

### Next Steps for Retrospective Branches

These branches already have their code merged into `main` (they were pushed directly before adopting the workflow). You can:

**Option 1: Create PRs for Documentation**
- Create PRs to document these were completed
- Add "Closes #N" to link to any related issues
- Provides audit trail

**Option 2: Keep as Historical Reference**
- Leave branches in place as feature documentation
- Don't create PRs since code is already in main
- Delete branches after documenting

**Recommended: Option 2** - The code is already in main and working. Creating PRs now would be artificial since there's nothing to review or merge.

---

## Best Practices

### 1. Keep Branches Small and Focused

- One feature per branch
- Aim for PRs < 500 lines changed
- Break large features into smaller PRs

### 2. Keep Branches Up to Date

```bash
# Regularly sync with main
git checkout feature/my-feature
git fetch origin
git rebase origin/main

# Or merge if you prefer
git merge origin/main
```

### 3. Write Good Commit Messages

```bash
# Good ✅
git commit -m "feat(api): Add rate limiting per API key

Implements sliding window rate limiter with configurable limits.
Adds X-RateLimit-* headers to all responses.

Closes #42"

# Bad ❌
git commit -m "fixed stuff"
```

### 4. Test Before Pushing

```bash
# Run tests locally
pytest tests/

# Run linting
flake8 src/

# Run type checking (if applicable)
mypy src/
```

### 5. Keep PRs Updated

- Address review comments promptly
- Update PR description if scope changes
- Rebase/merge to resolve conflicts

---

## Protected Branch Rules (Recommended)

### For `main` branch:

1. **Require pull request reviews**
   - At least 1 approval required
   - Dismiss stale reviews on new commits

2. **Require status checks**
   - All tests must pass
   - Linting checks must pass
   - Coverage threshold met (70%)

3. **Require branches to be up to date**
   - Must be current with main before merging

4. **Restrict who can push**
   - Only maintainers can push directly (for hotfixes)

### Setting up on GitHub:

1. Go to: Settings → Branches
2. Add rule for `main`
3. Enable protection options above

---

## Emergency Hotfix Process

For critical production issues:

```bash
# Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-security-fix

# Make minimal changes
git add <files>
git commit -m "hotfix: Fix critical security vulnerability"

# Push and create PR (fast-track review)
git push origin hotfix/critical-security-fix

# After approval, merge immediately
# Then merge back to any active feature branches
```

---

## Branch Cleanup

### Delete Local Branches

```bash
# Delete merged branch
git branch -d feature/my-feature

# Force delete unmerged branch
git branch -D feature/my-abandoned-feature
```

### Delete Remote Branches

```bash
# After PR is merged
git push origin --delete feature/my-feature

# Or on GitHub: "Delete branch" button after merging PR
```

### List Branches

```bash
# Local branches
git branch

# Remote branches
git branch -r

# All branches with commit info
git branch -av
```

---

## Common Commands Reference

```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main
git checkout feature/existing-feature

# Push new branch to remote
git push -u origin feature/new-feature

# Pull latest changes
git pull origin main

# Rebase on main
git rebase origin/main

# View branch status
git status
git branch -vv

# View commit history
git log --oneline --graph --all

# Create PR (with GitHub CLI)
gh pr create

# View PRs
gh pr list
```

---

## Troubleshooting

### Accidentally Committed to Main

```bash
# If not yet pushed
git reset --soft HEAD~1  # Undo commit, keep changes
git checkout -b feature/proper-branch
git commit -m "feat: Proper commit on feature branch"

# If already pushed (contact maintainer)
```

### Need to Update Feature Branch

```bash
# Option 1: Rebase (cleaner history)
git checkout feature/my-feature
git fetch origin
git rebase origin/main

# Option 2: Merge (safer if branch is shared)
git merge origin/main
```

### Conflicts During Merge

```bash
# Fix conflicts in files
git status  # See conflicted files
# Edit files, remove conflict markers
git add <resolved-files>
git rebase --continue  # Or git merge --continue
```

---

## Examples

### Example 1: Adding New Feature

```bash
# Start
git checkout main
git pull origin main
git checkout -b feature/add-grafana-dashboard

# Work
# ... make changes ...
git add src/dashboards/grafana_dashboard.py
git commit -m "feat(dashboards): Add Grafana dashboard integration"

# Push and PR
git push origin feature/add-grafana-dashboard
gh pr create --title "Add Grafana Dashboard Integration"

# After review and merge
git checkout main
git pull origin main
git branch -d feature/add-grafana-dashboard
```

### Example 2: Fixing Bug

```bash
# Start
git checkout main
git pull origin main
git checkout -b bugfix/api-rate-limit-leak

# Work
git add src/api/auth.py tests/api/test_auth.py
git commit -m "fix(api): Prevent rate limit counter leak on expired keys"

# Push and PR
git push origin bugfix/api-rate-limit-leak
gh pr create --title "Fix rate limit counter leak"
```

---

## Migration Complete ✅

**Status:** All feature work has been moved to feature branches.

**Current State:**
- ✅ 6 feature branches created and pushed
- ✅ All branches tracking remote
- ✅ `main` branch synced with remote
- ✅ Future work will use feature branch workflow

**Next Actions:**
- Continue using feature branches for all new work
- Consider setting up branch protection rules on GitHub
- Document completed features in PR descriptions

---

**Questions?** Review this document or check Git documentation at https://git-scm.com/doc

**Last Updated:** 2025-10-12
