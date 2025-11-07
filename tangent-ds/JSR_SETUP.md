# JSR (JavaScript Registry) Setup Guide

This document explains how the project is configured for dual publishing to both npm and JSR.

## Overview

This package is configured to be published to both:
- **npm**: Traditional Node.js package registry
- **JSR**: Modern JavaScript registry with first-class Deno and TypeScript support

## Configuration Files

### 1. `jsr.json`
Primary configuration for JSR publishing. Defines:
- Package name and version
- Export entry points
- Files to include/exclude from publication

### 2. `deno.json`
Deno runtime configuration that includes:
- Import maps for npm dependencies
- Lint and format settings
- Task definitions
- JSR package metadata

### 3. `package.json`
Enhanced with JSR-related scripts:
- `jsr:publish` - Publish to JSR
- `jsr:publish:dry-run` - Test publish without uploading
- `deno:lint` - Run Deno linter
- `deno:fmt` - Format code with Deno
- `deno:check` - Type check with Deno

## Prerequisites

To work with JSR, you need to install Deno:

```bash
# macOS/Linux
curl -fsSL https://deno.land/install.sh | sh

# Windows (PowerShell)
irm https://deno.land/install.ps1 | iex

# Using Homebrew (macOS)
brew install deno

# Using Cargo
cargo install deno --locked
```

After installation, add Deno to your PATH:
```bash
export DENO_INSTALL="$HOME/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"
```

## Testing with Deno

### 1. Lint Your Code
```bash
npm run deno:lint
# or directly:
deno lint src/
```

### 2. Format Your Code
```bash
npm run deno:fmt
# or directly:
deno fmt src/
```

### 3. Type Check
```bash
npm run deno:check
# or directly:
deno check src/index.js
```

## Publishing to JSR

### 1. Dry Run (Recommended First)
Test the publication process without actually publishing:

```bash
npm run jsr:publish:dry-run
# or directly:
deno publish --dry-run
```

This will:
- Validate your configuration
- Check file structure
- Show what would be published
- Report any errors or warnings

### 2. Actual Publication
Once the dry run succeeds:

```bash
npm run jsr:publish
# or directly:
deno publish
```

**Note**: You need to be authenticated with JSR. Run `deno publish --help` for authentication instructions.

## Import Maps for npm Dependencies

The `deno.json` file includes import maps that allow Deno to use npm packages:

```json
{
  "imports": {
    "ml-matrix": "npm:ml-matrix@^6.11.1",
    "simple-statistics": "npm:simple-statistics@^7.8.3",
    "@observablehq/plot": "npm:@observablehq/plot@^0.6.0"
  }
}
```

This ensures the package works seamlessly in Deno environments while maintaining npm compatibility.

## Dual Publishing Workflow

1. **Develop**: Write code using standard ES modules
2. **Test**: Run tests with both Node.js (vitest) and Deno
3. **Lint**: Check code quality with both ecosystems
   ```bash
   npm run test           # Node.js tests
   npm run deno:lint      # Deno linter
   ```
4. **Verify**: Test package structure
   ```bash
   npm run verify                 # npm package verification
   npm run jsr:publish:dry-run    # JSR package verification
   ```
5. **Publish**:
   ```bash
   npm publish              # Publish to npm
   npm run jsr:publish      # Publish to JSR
   ```

## Version Synchronization

Keep versions synchronized across:
- `package.json`
- `jsr.json`
- `deno.json`

**Tip**: Update the version in all three files before publishing to maintain consistency.

## JSR Package Usage

Once published, users can import your package in Deno:

```javascript
// Import from JSR
import { core, stats, ml } from "jsr:@tangent-to/ds";

// Or with version pinning
import { core } from "jsr:@tangent-to/ds@0.1.28";
```

## Troubleshooting

### Common Issues

1. **Import resolution errors**: Make sure all imports use explicit file extensions (`.js`)
2. **npm dependency issues**: Verify import maps in `deno.json` are correct
3. **Lint errors**: Run `deno lint` to identify and fix issues
4. **Type errors**: While using JavaScript, consider adding JSDoc comments for better type inference

### Getting Help

- JSR Documentation: https://jsr.io/docs
- Deno Documentation: https://deno.land/manual
- Report issues: https://github.com/tangent-to/tangent-ds/issues

## Migration Path to JSR

Since you plan to migrate to JSR in the long run:

1. **Phase 1** (Current): Dual publishing to npm and JSR
   - Maintain compatibility with both ecosystems
   - Test extensively on both platforms

2. **Phase 2**: Gradual migration
   - Encourage users to migrate to JSR
   - Keep npm packages updated but mark as legacy

3. **Phase 3**: JSR-first
   - Primary distribution via JSR
   - npm as secondary or deprecated

## Benefits of JSR

- **TypeScript-first**: Native TypeScript support without compilation
- **ESM-only**: Modern module system
- **Fast**: Optimized for performance
- **Secure**: Built-in security features
- **Better DX**: Improved developer experience with automatic documentation

