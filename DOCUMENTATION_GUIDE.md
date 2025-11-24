---
layout: default
title: Documentation Guide
nav_exclude: true
---

# Documentation Site Guide

This guide explains the structure and customization of the tangent/ds documentation site.

## Site Structure

The documentation uses [Jekyll](https://jekyllrb.com/) with the [just-the-docs](https://just-the-docs.com/) theme and is hosted on GitHub Pages.

### Custom Components

#### 1. Custom Header (`_includes/header_custom.html`)
- Displays site branding with logo and tagline
- Shows version badge
- Quick action buttons (Get Started, GitHub)
- Gradient design matching brand colors
- Fully responsive

#### 2. Custom Footer (`_includes/footer_custom.html`)
- Four-column layout with documentation links
- Quick links to all major sections
- Community resources
- Version information and badges
- License and copyright information

#### 3. Custom Head (`_includes/head_custom.html`)
- Custom fonts (Inter)
- Social media meta tags (Open Graph, Twitter Cards)
- Favicon
- Custom CSS for improved styling
- Syntax highlighting improvements

#### 4. Announcement Banner (`_includes/announcement.html`)
- Highlights new features and releases
- Gradient design matching header
- Update regularly for new releases

## Configuration

### Site Settings (`_config.yml`)

Key settings:
```yaml
title: "tangent/ds"
tagline: "data science toolkit for JavaScript"
version: "0.3.0"  # Update for each release
url: "https://tangent-to.github.io"
baseurl: "/ds"
repository: tangent-to/ds
```

### Navigation Order

Pages use `nav_order` in front matter:
1. Home
2. Getting Started
3. Tutorials
4. API Reference
5. Examples
6. Statistics
7. Machine Learning
8. Multivariate Analysis
9. Visualization
10. Core Utilities

## Content Organization

### Main Pages
- `index.md` - Homepage with feature overview
- `getting-started.md` - Installation and quick start
- `tutorials.md` - Learning resources
- `api-reference.md` - API documentation
- `examples.md` - Code examples

### Module Documentation
- `stats.md` - Statistics module
- `ml.md` - Machine learning module
- `mva.md` - Multivariate analysis module
- `plot.md` - Visualization module
- `core.md` - Core utilities

### Collections
- `_tutorials/` - Tutorial pages
- `_api/` - Detailed API documentation

## Styling

### Colors
- **Primary**: `#667eea` (purple-blue)
- **Primary Dark**: `#5568d3`
- **Secondary**: `#764ba2` (purple)
- **Background**: `#f6f8fa` (light gray)

### Typography
- **Body**: Inter, system fonts
- **Code**: Monospace
- **Headings**: Inter Bold

## Updating the Documentation

### For New Releases

1. **Update version** in `_config.yml`:
   ```yaml
   version: "0.3.1"
   ```

2. **Update announcement banner** in `_includes/announcement.html`

3. **Add to changelog** if you have one

### Adding New Pages

1. Create markdown file in `docs/` directory
2. Add front matter:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   nav_order: 11
   description: "Page description"
   permalink: /your-page
   ---
   ```

3. Content using Markdown + Jekyll Liquid

### Adding Tutorials

1. Create file in `docs/_tutorials/`
2. Add front matter:
   ```yaml
   ---
   title: Tutorial Title
   description: "Brief description"
   ---
   ```

3. Will automatically appear under Tutorials

## Local Development

### Prerequisites
- Ruby 2.7+
- Bundler

### Setup
```bash
cd docs
bundle install
bundle exec jekyll serve
```

Visit `http://localhost:4000/ds/`

### Live Reload
Jekyll watches for changes automatically. Refresh browser to see updates.

## Deployment

The site automatically deploys via GitHub Actions when changes are pushed to the main branch.

### Manual Deployment
GitHub Pages builds automatically from the `docs/` directory.

## Best Practices

1. **Keep navigation organized** - Use consistent `nav_order`
2. **Add descriptions** - Help with SEO and clarity
3. **Use permalinks** - Clean URLs
4. **Test locally** - Before pushing
5. **Mobile-first** - Ensure responsive design
6. **Accessibility** - Use semantic HTML, alt text
7. **Update announcement** - For each release

## Troubleshooting

### Build Failures
- Check Jekyll syntax in front matter
- Validate Liquid template syntax
- Ensure all includes exist

### Navigation Issues
- Check `nav_order` numbers
- Verify `parent` and `has_children` settings
- Ensure permalink URLs are correct

### Styling Issues
- Clear browser cache
- Check CSS specificity
- Verify include files are loading

## Resources

- [Just the Docs Documentation](https://just-the-docs.com/)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)

## Support

For documentation issues, please [open an issue](https://github.com/tangent-to/ds/issues) on GitHub.
