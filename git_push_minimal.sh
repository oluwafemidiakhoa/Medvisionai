
#!/bin/bash
# Script for a minimal Git push focusing only on important files

echo "=== Minimal Git Push ==="

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Stage only critical files
echo "Adding only essential files..."
git add app.py config.py requirements.txt .replit .gitignore .gitattributes

# Commit the changes
echo "Committing essential changes..."
git commit -m "Update core application files"

# Push only the current branch with verbose output
echo "Pushing only current branch with verbose output..."
git push -v origin "$CURRENT_BRANCH"

echo "=== Done ==="
