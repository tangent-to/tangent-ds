# Publishing Guide

This package is configured for dual publishing to npm and JSR (JavaScript Registry).

## Quick Start

### Publishing to npm
```bash
# 1. Update version
npm version patch|minor|major

# 2. Build and test
npm run test:run
npm run build:browser
npm run verify

# 3. Publish
npm publish
```

### Publishing to JSR
```bash
# 1. Test first (requires Deno)
npm run jsr:publish:dry-run

# 2. Publish
npm run jsr:publish
```

## Version Management

**IMPORTANT**: Keep versions synchronized across all configuration files:

1. Update in `package.json`
2. Update in `jsr.json`
3. Update in `deno.json`

Script to update all at once:
```bash
NEW_VERSION="0.1.29"
sed -i "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" package.json
sed -i "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" jsr.json
sed -i "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" deno.json
```

## Pre-Publication Checklist

### For npm:
- [ ] All tests pass (`npm run test:run`)
- [ ] Build succeeds (`npm run build:browser`)
- [ ] Package verification passes (`npm run verify`)
- [ ] README.md is up to date
- [ ] CHANGELOG updated (if applicable)

### For JSR (additional checks):
- [ ] Deno lint passes (`npm run deno:lint`)
- [ ] Deno format check passes (`npm run deno:fmt --check`)
- [ ] Deno type check passes (`npm run deno:check`)
- [ ] Dry run succeeds (`npm run jsr:publish:dry-run`)

## Complete Publishing Workflow

```bash
# 1. Make sure everything is committed
git status

# 2. Run tests
npm run test:run

# 3. Update versions (all three files)
# Edit package.json, jsr.json, deno.json

# 4. Build
npm run build:browser

# 5. Verify npm package
npm run verify

# 6. Test JSR package (if Deno is available)
npm run deno:lint
npm run deno:check
npm run jsr:publish:dry-run

# 7. Commit version bump
git add .
git commit -m "chore: bump version to X.Y.Z"

# 8. Tag release
git tag vX.Y.Z

# 9. Push to GitHub
git push && git push --tags

# 10. Publish to npm
npm publish

# 11. Publish to JSR
npm run jsr:publish
```

## Configuration Files

| File | Purpose |
|------|---------|
| `package.json` | npm configuration |
| `jsr.json` | JSR package metadata |
| `deno.json` | Deno runtime & JSR configuration |
| `.npmignore` | Files excluded from npm package |
| `.jsrignore` | Files excluded from JSR package |

## Package Structure

```
tangent-ds/
├── src/              # Source files (published to both)
│   ├── core/
│   ├── stats/
│   ├── ml/
│   ├── mva/
│   └── plot/
├── dist/             # Built files (npm only)
├── tests/            # Test files (excluded)
├── examples/         # Examples (excluded)
├── package.json      # npm config
├── jsr.json          # JSR config
└── deno.json         # Deno config
```

## Usage Examples

### npm (Node.js/Browser)
```javascript
import { core, stats, ml } from '@tangent.to/ds';
```

### JSR (Deno)
```javascript
import { core, stats, ml } from 'jsr:@tangent-to/ds';
```

### JSR with version pinning
```javascript
import { core } from 'jsr:@tangent-to/ds@0.1.28';
```

## Troubleshooting

### npm publish fails with 403
- Check if you're logged in: `npm whoami`
- Login if needed: `npm login`
- Verify package name is not taken

### JSR publish fails
- Make sure Deno is installed: `deno --version`
- Run dry-run first: `npm run jsr:publish:dry-run`
- Check JSR documentation: https://jsr.io/docs/publishing-packages

### Version mismatch errors
- Ensure all three config files have the same version
- Use the version sync script above

## Resources

- [npm Publishing Guide](https://docs.npmjs.com/packages-and-modules/contributing-packages-to-the-registry)
- [JSR Documentation](https://jsr.io/docs)
- [Deno Manual](https://deno.land/manual)

## Support

- Issues: https://github.com/tangent-to/tangent-ds/issues
- Discussions: https://github.com/tangent-to/tangent-ds/discussions
