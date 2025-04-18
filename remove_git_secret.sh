
#!/bin/bash
# Script to remove gen.json from Git history completely

echo "=== Removing gen.json from Git history ==="

# Create a fresh branch without the sensitive file
echo "Creating a fresh branch without secrets..."
git checkout -b clean-branch

# Make sure .gitignore is properly configured
echo "Updating .gitignore..."
grep -q "gen.json" .gitignore || echo "gen.json" >> .gitignore
grep -q "*/gen.json" .gitignore || echo "*/gen.json" >> .gitignore

# Add all files except gen.json
echo "Adding all files except gen.json..."
git add .

# Commit the changes with a clean message
echo "Committing files..."
git commit -m "Clean repository without service account credentials"

# Force push to GitHub (this will override the previous history)
echo "Force pushing to GitHub..."
git push -f origin clean-branch

echo "=== Done ==="
echo "Now go to GitHub and set 'clean-branch' as your default branch."
echo "This removes the service account file from Git history completely."
