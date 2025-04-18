
#!/bin/bash
# Script to push code without using Git LFS

echo "=== Pushing code without Git LFS ==="

# Commit the .gitattributes change
echo "Committing .gitattributes changes..."
git add .gitattributes
git commit -m "Disable Git LFS tracking completely"

# Push to a new branch
echo "Pushing to origin..."
git push -u origin fix-lfs-branch

echo "=== Done ==="
echo "If successful, make fix-lfs-branch your default branch in GitHub repository settings."
