# Documentation Development

This directory contains the documentation for @tangent.to/ds, built with Jekyll and deployed to GitHub Pages.

## Local Development

### Prerequisites

- Ruby 2.7 or higher
- Bundler

### Setup

```bash
# Install Ruby dependencies
cd tangent-ds/docs
bundle install

# Generate API docs from source
cd ..
npm run docs:generate

# Serve locally with Jekyll
cd docs
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000/tangent-ds/`

## File Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── index.md             # Homepage
├── API.md               # Full API reference (generated)
├── core.md              # Core module docs (generated)
├── stats.md             # Stats module docs (generated)
├── ml.md                # ML module docs (generated)
├── mva.md               # MVA module docs (generated)
├── plot.md              # Plot module docs (generated)
├── Gemfile              # Ruby dependencies
└── README_DOCS.md       # This file
```

## Regenerating Documentation

Documentation is automatically generated from source code JSDoc comments:

```bash
npm run docs:generate
```

This creates/updates the markdown files in `docs/`.

## Deployment

Documentation is automatically deployed to GitHub Pages when:

1. **On release**: When a version tag is pushed (e.g., `v0.1.29`)
2. **On changes**: When documentation files are updated on `main` branch

The deployment is handled by GitHub Actions workflows:
- `.github/workflows/publish.yml` - Deploys on release
- `.github/workflows/docs.yml` - Deploys on documentation changes

## Theme

We use the [Cayman theme](https://github.com/pages-themes/cayman) for GitHub Pages.

To change the theme, edit `_config.yml`:

```yaml
theme: jekyll-theme-cayman  # or another theme
```

Available GitHub Pages themes:
- `jekyll-theme-cayman`
- `jekyll-theme-minimal`
- `jekyll-theme-slate`
- `jekyll-theme-architect`
- `jekyll-theme-tactile`
- `minima`

## Live Site

The documentation is published at: https://tangent-to.github.io/tangent-ds/
