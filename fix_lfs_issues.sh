
#!/bin/bash
# Script to fix Git LFS issues

echo "=== Fixing Git LFS issues ==="

# Make sure Git LFS is installed
echo "Making sure Git LFS is installed..."
git lfs install

# Remove problematic files from Git tracking
echo "Removing problematic LFS files from Git tracking..."
git rm --cached assets/radvisionai-hero.jpeg assets/demo.png || echo "Files already removed or not tracked"

# Add missing LFS files to .gitignore if needed
echo "Checking .gitignore..."
grep -q "assets/radvisionai-hero.jpeg" .gitignore || echo "assets/radvisionai-hero.jpeg" >> .gitignore
grep -q "assets/demo.png" .gitignore || echo "assets/demo.png" >> .gitignore

# Commit the changes
echo "Committing changes..."
git add .gitignore .gitattributes
git commit -m "Fix Git LFS tracking issues"

echo "=== Done ==="
echo "Now try pushing with: git push origin fix-lfs-branch"
