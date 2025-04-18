
#!/bin/bash
# Script to safely push to Git

echo "=== Preparing to push to Git ==="

# Make sure service account credentials are not tracked
echo "Checking for sensitive files..."
if git ls-files | grep -q "gen.json"; then
  echo "❌ WARNING: gen.json is still being tracked by Git."
  echo "This file contains credentials and should not be committed."
  
  # Remove from Git tracking
  echo "Removing gen.json from Git tracking..."
  git rm --cached gen.json
  
  # Make sure it's in .gitignore
  if ! grep -q "gen.json" .gitignore; then
    echo "Adding gen.json to .gitignore..."
    echo "gen.json" >> .gitignore
    git add .gitignore
  fi
  
  git commit -m "Remove service account credentials from Git tracking"
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Check if there are any changes to commit
if [ -n "$(git status --porcelain)" ]; then
  echo "You have uncommitted changes. Would you like to commit them? (y/n)"
  read -r COMMIT_RESPONSE
  if [ "$COMMIT_RESPONSE" = "y" ]; then
    echo "Enter a commit message:"
    read -r COMMIT_MESSAGE
    git add .
    git commit -m "$COMMIT_MESSAGE"
  else
    echo "Skipping commit of pending changes."
  fi
fi

# Push changes
echo "Pushing to origin/$CURRENT_BRANCH..."
git push -u origin "$CURRENT_BRANCH"

PUSH_STATUS=$?
if [ $PUSH_STATUS -eq 0 ]; then
  echo "✅ Successfully pushed to origin/$CURRENT_BRANCH"
else
  echo "❌ Push failed with status $PUSH_STATUS"
  echo "If you're seeing a 'secret detected' error, make sure all sensitive files like gen.json"
  echo "are removed from Git tracking and added to .gitignore."
  echo "Try running the clean_repository.sh script first."
fi

echo "=== Done ==="
