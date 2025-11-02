# Release Process

This document describes the automated release process for @tangent.to/ds.

## Overview

The project uses GitHub Actions to automate publishing when a version tag is pushed. The workflow handles:

1. Running the full test suite
2. Building the browser bundle
3. Publishing to npm with provenance
4. Deploying the documentation site to GitHub Pages

## Workflow Trigger

The workflow is triggered when a tag matching the pattern `v*` is pushed (e.g., `v0.7.0`, `v1.0.0`).

## Prerequisites

Before the first release, ensure the following secrets are configured in GitHub repository settings:

- `NPM_TOKEN`: An npm access token with publish permissions
  - Go to npmjs.com → Access Tokens → Generate New Token
  - Select "Automation" type for CI/CD use
  - Add to GitHub repo: Settings → Secrets and variables → Actions → New repository secret

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions (no setup needed)

## Release Steps

### 1. Prepare the Release

```bash
# Ensure you're on the main branch and up to date
git checkout main
git pull origin main

# Navigate to the package directory
cd tangent-ds

# Run tests to ensure everything works
npm run test:run

# Build and verify the package
npm run build:browser
npm run verify
```

### 2. Update Version

**For the first release** (0.1.0), version is already set in package.json.

**For subsequent releases**, use npm version:

```bash
cd tangent-ds/

npm version patch  # Bug fixes: 0.1.0 → 0.1.1
npm version minor  # New features: 0.1.0 → 0.2.0
npm version major  # Breaking changes: 0.1.0 → 1.0.0

cd ..
```

### 3. Create and Push the Tag

**For first release:**

```bash
# Create the tag
# git tag v0.1.0

# Push code and tag
git push origin main --tags
```

**For subsequent releases** (npm version creates tag automatically):

```bash
# After npm version patch/minor/major
git push origin main --tags
```

### 4. Monitor the Workflow

1. Go to GitHub repository → Actions tab
2. Watch the "Publish Package and Site" workflow run
3. The workflow includes two jobs:
   - **publish-npm**: Tests, builds, and publishes to npm
   - **deploy-site**: Builds and deploys documentation to GitHub Pages

### 5. Verify the Release

After the workflow completes successfully:

**Check npm:**
```bash
npm view @tangent.to/ds version
# Should show the new version
```

**Check GitHub Pages:**
- Visit your GitHub Pages URL (e.g., `https://yourusername.github.io/tangent-ds`)
- Verify the documentation is updated

**Check GitHub Releases:**
- Consider creating a GitHub Release from the tag
- Document the changes and features in the release notes

## Workflow Details

### publish-npm Job

This job runs on `ubuntu-latest` and performs:

1. Checkout code
2. Setup Node.js 20 with npm registry
3. Install dependencies (`npm ci`)
4. Run tests (`npm run test:run`)
5. Build browser bundle (`npm run build:browser`)
6. Verify package structure (`npm run verify`)
7. Publish to npm with provenance (`npm publish --provenance --access public`)

### deploy-site Job

This job runs after `publish-npm` succeeds and performs:

1. Checkout code
2. Setup Node.js 20
3. Install docs dependencies
4. Build documentation site
5. Deploy to GitHub Pages using peaceiris/actions-gh-pages

## Troubleshooting

### Workflow Fails on npm Publish

- **Error: 403 Forbidden**
  - Check that `NPM_TOKEN` secret is set correctly
  - Verify the token has publish permissions
  - Ensure you have access to publish the `@tangent.to` scope

- **Error: Version already exists**
  - The version in package.json was already published
  - Update to a new version number and create a new tag

### Workflow Fails on Tests

- The workflow will not publish if tests fail
- Fix the tests locally and create a new version tag

### Site Deployment Fails

- Check that GitHub Pages is enabled in repository settings
- Verify the `GITHUB_TOKEN` has write permissions
- Ensure the build directory exists and contains the built site

## Manual Publishing (Fallback)

If the automated workflow fails, you can publish manually:

```bash
cd tangent-ds

# Publish to npm
npm publish --access public

# Deploy documentation
cd docs
npm run build
npm run deploy
```

## Best Practices

1. **Always test before releasing**: Run the full test suite locally
2. **Follow semantic versioning**:
   - Patch: Bug fixes
   - Minor: New features (backward compatible)
   - Major: Breaking changes
3. **Update documentation**: Ensure docs are current before releasing
4. **Changelog**: Consider maintaining a CHANGELOG.md file
5. **Release notes**: Document changes in GitHub Releases

## Workflow File

The workflow is defined in [.github/workflows/publish.yml](.github/workflows/publish.yml).
