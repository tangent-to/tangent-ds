# capture your changes
git add .
git commit -m "<meaningful summary>"
git status

# bump and tag (npm creates commit + tag now)
cd tangent-ds
npm version patch   # or minor / major
cd ..

# publish
git push origin main --follow-tags