#!/bin/bash
# CI/CD Setup Script for SRE-Analytics
# This script sets up the local development environment for CI/CD

set -e  # Exit on error

echo "🚀 Setting up CI/CD for SRE-Analytics..."
echo ""

# Check Python version
echo "✅ Checking Python version..."
python_version=$(python3 --version)
echo "   Found: $python_version"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install pre-commit
echo ""
echo "✅ Installing pre-commit..."
pip install pre-commit

# Install code quality tools
echo ""
echo "✅ Installing code quality tools..."
pip install black flake8 isort mypy bandit safety pytest-cov

# Install pre-commit hooks
echo ""
echo "✅ Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files (optional, can be slow)
echo ""
read -p "Do you want to run pre-commit on all files now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "✅ Running pre-commit on all files..."
    pre-commit run --all-files || true
fi

# Check if git is initialized
echo ""
if [ -d .git ]; then
    echo "✅ Git repository detected"

    # Check current branch
    current_branch=$(git branch --show-current)
    echo "   Current branch: $current_branch"

    # Check if remote is set
    if git remote get-url origin &> /dev/null; then
        remote_url=$(git remote get-url origin)
        echo "   Remote URL: $remote_url"
    else
        echo "⚠️  No remote URL set. Add with: git remote add origin <url>"
    fi
else
    echo "⚠️  Not a git repository. Initialize with: git init"
fi

echo ""
echo "✅ CI/CD setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Review .github/workflows/ for GitHub Actions configuration"
echo "   2. Set up GitHub secrets (see docs/CI_CD_GUIDE.md)"
echo "   3. Configure Codecov integration"
echo "   4. Enable branch protection rules"
echo "   5. Test pre-commit hooks: git commit"
echo ""
echo "📚 Documentation: docs/CI_CD_GUIDE.md"
echo ""
