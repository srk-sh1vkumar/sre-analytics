# 🔄 Applying CI/CD to Other Projects

**Question:** Is the CI/CD pipeline set for all projects?

**Answer:** No, this CI/CD pipeline is currently set up **only for sre-analytics**. However, you can easily apply it to your other projects!

---

## 📊 Current Status

### ✅ **sre-analytics** - CI/CD Configured
- GitHub Actions workflows
- Pre-commit hooks
- Code quality tools
- Coverage reporting
- Automated deployment

### 🔲 **Other Projects** - Not Yet Configured

Your other projects (like ecommerce-microservices) don't have CI/CD yet, but you can add it easily!

---

## 🚀 How to Apply CI/CD to Other Projects

### **Option 1: Copy Configuration Files** (Recommended for Python projects)

For **ecommerce-microservices** or other Python projects:

```bash
# 1. Navigate to your other project
cd /Users/shiva/Projects/ecommerce-microservices

# 2. Copy CI/CD files from sre-analytics
cp -r /Users/shiva/Projects/sre-analytics/.github .
cp /Users/shiva/Projects/sre-analytics/.pre-commit-config.yaml .
cp /Users/shiva/Projects/sre-analytics/.flake8 .
cp /Users/shiva/Projects/sre-analytics/.yamllint .
cp /Users/shiva/Projects/sre-analytics/.codecov.yml .

# 3. Copy or merge pyproject.toml configurations
# (Be careful not to overwrite existing project-specific configs)

# 4. Install pre-commit
pip install pre-commit
pre-commit install

# 5. Test and commit
git add .github/ .pre-commit-config.yaml .flake8 .yamllint .codecov.yml
git commit -m "feat(cicd): Add CI/CD pipeline configuration"
git push
```

---

### **Option 2: Adapt for Java/Spring Boot Projects**

For **ecommerce-microservices** (if it's Java-based), you need different tools:

```yaml
# .github/workflows/test.yml for Java
name: Java CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up JDK 17
        uses: actions/setup-java@v3
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Build with Maven
        run: mvn clean install

      - name: Run tests
        run: mvn test

      - name: Generate coverage report
        run: mvn jacoco:report

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./target/site/jacoco/jacoco.xml
```

**Java CI/CD Tools:**
- **Build:** Maven or Gradle
- **Testing:** JUnit, TestNG
- **Code Quality:** SonarQube, Checkstyle, PMD
- **Coverage:** JaCoCo
- **Static Analysis:** SpotBugs, FindBugs

---

### **Option 3: Use a Template Repository**

Create a template that can be reused:

```bash
# 1. Create a new repository: cicd-templates
# 2. Add language-specific folders:
#    - cicd-templates/python/
#    - cicd-templates/java/
#    - cicd-templates/javascript/
# 3. Store reusable workflow files in each

# Then copy templates to new projects:
cp -r cicd-templates/python/.github my-new-project/
```

---

## 📋 Project-Specific Adaptations Needed

When copying CI/CD to another project, you'll need to modify:

### 1. **Project Name**
```yaml
# In .github/workflows/deploy.yml
# Change:
images: ${{ secrets.DOCKER_USERNAME }}/sre-analytics
# To:
images: ${{ secrets.DOCKER_USERNAME }}/your-project-name
```

### 2. **Programming Language**
- **Python:** Use pytest, black, flake8
- **Java:** Use Maven/Gradle, JUnit, Checkstyle
- **JavaScript/Node.js:** Use Jest, ESLint, Prettier
- **Go:** Use go test, golint, gofmt

### 3. **Dependencies**
```yaml
# Python:
pip install -r requirements.txt

# Java:
mvn clean install

# Node.js:
npm install

# Go:
go mod download
```

### 4. **Test Commands**
```yaml
# Python:
pytest --cov=src

# Java:
mvn test

# Node.js:
npm test

# Go:
go test ./...
```

### 5. **Build Commands**
```yaml
# Python:
pip install .

# Java:
mvn package

# Node.js:
npm run build

# Go:
go build
```

---

## 🎯 Recommended Rollout Strategy

### **Phase 1: Add CI/CD to sre-analytics** (Current) ✅
- Test the pipeline
- Fix any issues
- Document lessons learned

### **Phase 2: Add CI/CD to Main Projects**
Priority order for your projects:

1. **ecommerce-microservices** (High Priority)
   - Most active development
   - Multiple services to test
   - Docker deployment

2. **Other Python projects** (Medium Priority)
   - Copy sre-analytics CI/CD directly
   - Minimal modifications needed

3. **Experimental/Learning projects** (Low Priority)
   - Add CI/CD as needed
   - Good for testing new workflows

---

## 🔧 Customization Guide

### **For Each Project, Customize:**

#### **1. Coverage Thresholds**
```yaml
# Adjust based on project maturity
pytest --cov=src --cov-report=xml --fail-under=70  # New projects: 70%
pytest --cov=src --cov-report=xml --fail-under=80  # Mature projects: 80%+
```

#### **2. Test Timeout**
```yaml
# For fast unit tests:
timeout-minutes: 5

# For integration tests:
timeout-minutes: 15

# For E2E tests:
timeout-minutes: 30
```

#### **3. Python Versions to Test**
```yaml
# For production projects:
python-version: ['3.9', '3.10', '3.11']

# For newer projects:
python-version: ['3.11', '3.12']

# For legacy projects:
python-version: ['3.8', '3.9']
```

#### **4. Deployment Strategy**
```yaml
# Option A: Deploy on every push to main (staging)
on:
  push:
    branches: [main]

# Option B: Deploy only on version tags (production)
on:
  push:
    tags:
      - 'v*'

# Option C: Manual approval (most projects)
environment:
  name: production
  # Requires manual approval in GitHub
```

---

## 💡 Project-Specific Examples

### **Example 1: ecommerce-microservices (Java/Spring Boot)**

```yaml
# .github/workflows/test.yml
name: Microservices CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-services:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [user-service, product-service, order-service, payment-service]

    steps:
      - uses: actions/checkout@v3

      - name: Set up JDK 17
        uses: actions/setup-java@v3
        with:
          java-version: '17'

      - name: Test ${{ matrix.service }}
        working-directory: ./${{ matrix.service }}
        run: |
          mvn clean test
          mvn jacoco:report

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./${{ matrix.service }}/target/site/jacoco/jacoco.xml
          flags: ${{ matrix.service }}
```

### **Example 2: React Frontend**

```yaml
# .github/workflows/test.yml
name: Frontend CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests
        run: npm test -- --coverage

      - name: Build
        run: npm run build

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 🔐 Shared GitHub Secrets

For multiple projects in the same organization, set up **Organization Secrets**:

1. Go to: GitHub Organization → Settings → Secrets and variables → Actions
2. Add organization-wide secrets:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `CODECOV_TOKEN`
   - `SLACK_WEBHOOK_URL`

Then all projects can use the same secrets!

---

## 📊 Monitoring Multiple Projects

### **Option 1: GitHub Organization Dashboard**
- View all workflow runs across projects
- Filter by status (passing/failing)
- Monitor deployment frequency

### **Option 2: Codecov Organization**
- Unified coverage dashboard
- Compare coverage across projects
- Track trends over time

### **Option 3: Custom Dashboard**
Use GitHub API to build a custom CI/CD dashboard:
```python
# Fetch workflow status for all repos
repos = ['sre-analytics', 'ecommerce-microservices', 'project3']
for repo in repos:
    status = github_api.get_workflow_status(repo)
    print(f"{repo}: {status}")
```

---

## ✅ Next Steps

### **For sre-analytics (This Project):**
1. ✅ Complete CI/CD setup
2. ✅ Test with first PR
3. ✅ Merge to main
4. ✅ Monitor first deployment

### **For Other Projects:**
1. 🔲 Identify which projects need CI/CD
2. 🔲 Adapt workflows for each language
3. 🔲 Set up organization secrets
4. 🔲 Roll out incrementally (one project at a time)

---

## 🎓 Learning Resources

- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **Pre-commit Framework:** https://pre-commit.com
- **Codecov Docs:** https://docs.codecov.com
- **CI/CD Best Practices:** https://github.com/features/actions

---

## 📞 Summary

**Your Question:** Is the CI/CD pipeline set for all projects?

**Answer:**
- ✅ **sre-analytics:** CI/CD is being set up now (almost complete!)
- 🔲 **ecommerce-microservices:** No CI/CD yet (can be added easily)
- 🔲 **Other projects:** No CI/CD yet

**Recommendation:**
1. Finish sre-analytics CI/CD first (today)
2. Test and validate it works (this week)
3. Apply to ecommerce-microservices next (next week)
4. Roll out to other projects incrementally

This approach ensures you have a working template before applying it everywhere!

---

**Last Updated:** 2025-10-13
**Status:** sre-analytics CI/CD in progress
