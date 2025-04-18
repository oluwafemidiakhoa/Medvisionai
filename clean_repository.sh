
#!/bin/bash
# Script to clean repository of secrets and prepare for push

echo "=== Cleaning Repository of Secrets ==="

# Create a .gitignore entry for gen.json if it doesn't exist
echo "Updating .gitignore to exclude service account file..."
grep -q "gen.json" .gitignore || echo "gen.json" >> .gitignore

# Remove gen.json from git tracking without deleting the file
echo "Removing gen.json from git tracking..."
git rm --cached gen.json 2>/dev/null || echo "gen.json not in git index"

# Move contents to Replit Secrets
echo "Moving service account to Replit Secrets..."
python move_credentials_to_secrets.py

# Commit these changes
echo "Committing changes..."
git add .gitignore
git commit -m "Remove service account credentials from repository"

echo "=== Done ==="
echo "Now try pushing again with: git push origin fix-lfs-branch"
echo ""
echo "IMPORTANT: After pushing, go to the Replit Secrets tab to securely store your credentials"
echo "as described in the output above."
