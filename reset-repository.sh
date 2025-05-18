#!/bin/bash
# Script to create a new repository without the bloated history

# Step 1: Create a new branch at the current state without history
echo "Creating a new branch without history..."
git checkout --orphan temp_branch

# Step 2: Add all files (will respect your .gitignore)
git add .

# Step 3: Commit the current state
git commit -m "Initial commit"

# Step 4: Delete the old branch (typically main or master)
git branch -D main  # Change to 'master' if that's your main branch name

# Step 5: Rename the temporary branch to be the main branch
git branch -m main

# Step 6: Force push to remote
echo "Ready to push the new repository without history."
echo "Run 'git push -f origin main' when you're ready."
