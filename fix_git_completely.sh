
#!/bin/bash
# Script to completely fix Git issues - both LFS and credentials

echo "=== Fixing Git Issues Completely ==="

# 1. Install and configure Git LFS properly
echo "Setting up Git LFS..."
git lfs install

# 2. Make sure gen.json is properly gitignored
echo "Ensuring credentials are gitignored..."
grep -q "gen.json" .gitignore || echo "gen.json" >> .gitignore
grep -q "*.json" .gitattributes || echo "*.json filter=lfs diff=lfs merge=lfs -text" >> .gitattributes

# 3. Create a completely fresh branch without history
echo "Creating a fresh branch without history..."
FRESH_BRANCH="clean-medvision"
git checkout --orphan $FRESH_BRANCH

# 4. Add all files except sensitive ones
echo "Adding all non-sensitive files..."
git rm -rf --cached .
git add .
git reset -- gen.json

# 5. Commit the initial clean state
echo "Committing clean repository state..."
git commit -m "Initial clean repository state (without credentials)"

# 6. Try to identify and fix LFS issues
echo "Fetching LFS objects..."
git lfs fetch --all

# 7. Push with force to override the old history
echo "Force pushing clean branch to GitHub..."
git push -f origin $FRESH_BRANCH

echo "=== Done ==="
echo "If successful, make $FRESH_BRANCH your default branch in GitHub repository settings."
echo "Then delete the old branches that contained credentials."
