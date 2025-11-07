# JSR Testing Checklist

Quick reference for testing JSR compatibility before publishing.

## Prerequisites Check

```bash
# Verify Deno is installed
deno --version

# Should output something like:
# deno 1.x.x (release, x86_64-unknown-linux-gnu)
# v8 x.x.x.x
# typescript 5.x.x
```

## Pre-Publication Checklist

### 1. Lint Check ✓
```bash
npm run deno:lint
```

**Expected**: No errors or warnings
**Common issues**:
- Missing file extensions in imports
- Unused variables
- Console.log statements (warnings)

### 2. Format Check ✓
```bash
npm run deno:fmt --check
```

**Expected**: All files properly formatted
**To fix**: Run `npm run deno:fmt` to auto-format

### 3. Type Check ✓
```bash
npm run deno:check
```

**Expected**: No type errors
**Common issues**:
- Missing type imports
- Incorrect type annotations in JSDoc

### 4. Import Validation ✓

Check all imports have explicit extensions:

```bash
# Bad:
import { foo } from './bar'

# Good:
import { foo } from './bar.js'
```

Verify with:
```bash
grep -r "from '[^']*[^.js]'" src/ || echo "All imports OK"
```

### 5. JSR Dry Run ✓
```bash
npm run jsr:publish:dry-run
```

**Expected output**:
```
Check file:///path/to/tangent-ds/src/index.js
Checking for slow types in the public API...
Check file:///path/to/tangent-ds/src/core/index.js
...
Simulating publish of @tangent-to/ds@0.1.28
...
```

**Red flags**:
- ❌ Missing files
- ❌ Slow types detected
- ❌ Invalid configuration
- ⚠️ Warnings about types

### 6. Package Structure ✓

Verify included files:
```bash
cat jsr.json | grep -A 10 "publish"
```

Should include:
- ✓ Source files (src/)
- ✓ README.md
- ✓ LICENSE
- ✗ No test files
- ✗ No build artifacts
- ✗ No node_modules

### 7. Version Consistency ✓

Check versions match:
```bash
echo "package.json:" && grep '"version"' package.json
echo "jsr.json:" && grep '"version"' jsr.json
echo "deno.json:" && grep '"version"' deno.json
```

All three should show the same version.

### 8. Dependency Check ✓

Verify npm dependencies are mapped correctly in `deno.json`:
```bash
# Check deno.json imports section
cat deno.json | grep -A 5 '"imports"'
```

Should match dependencies in `package.json`.

## Testing the Published Package

After publishing to JSR, test the package:

### 1. Test Import
```bash
deno eval "import * as ds from 'jsr:@tangent-to/ds'; console.log(ds.core)"
```

### 2. Test Functionality
Create a test file `test_jsr.js`:
```javascript
import { core, stats } from "jsr:@tangent-to/ds";

console.log("Testing core:", typeof core);
console.log("Testing stats:", typeof stats);
```

Run it:
```bash
deno run test_jsr.js
```

## Common Errors and Solutions

### Error: "Could not resolve import"
**Cause**: Missing file extension in import statement
**Fix**: Add `.js` extension to all relative imports

### Error: "Slow types detected"
**Cause**: Complex type inference without explicit annotations
**Fix**: Add JSDoc type comments or create `.d.ts` files

### Error: "Failed to publish: already exists"
**Cause**: Version already published
**Fix**: Increment version in all config files

### Error: "Invalid package name"
**Cause**: Package name doesn't match JSR naming conventions
**Fix**: Use format `@scope/package-name` (lowercase, hyphens only)

### Error: "npm dependency not found"
**Cause**: Import map missing or incorrect
**Fix**: Update `imports` section in `deno.json`

## Automated Testing Script

Save as `test-jsr.sh`:

```bash
#!/bin/bash

echo "🔍 JSR Compatibility Test Suite"
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

errors=0

# Test 1: Deno installed
echo -n "1. Checking Deno installation... "
if command -v deno &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Please install Deno first"
    exit 1
fi

# Test 2: Lint
echo -n "2. Running Deno lint... "
if deno lint src/ &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    errors=$((errors + 1))
fi

# Test 3: Format check
echo -n "3. Checking format... "
if deno fmt --check src/ &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    errors=$((errors + 1))
fi

# Test 4: Type check
echo -n "4. Type checking... "
if deno check src/index.js &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    errors=$((errors + 1))
fi

# Test 5: Version consistency
echo -n "5. Checking version consistency... "
v1=$(grep '"version"' package.json | cut -d'"' -f4)
v2=$(grep '"version"' jsr.json | cut -d'"' -f4)
v3=$(grep '"version"' deno.json | cut -d'"' -f4)
if [ "$v1" = "$v2" ] && [ "$v2" = "$v3" ]; then
    echo -e "${GREEN}✓${NC} (v$v1)"
else
    echo -e "${RED}✗${NC}"
    echo "   package.json: $v1"
    echo "   jsr.json: $v2"
    echo "   deno.json: $v3"
    errors=$((errors + 1))
fi

# Test 6: Dry run
echo -n "6. JSR dry run publish... "
if deno publish --dry-run --allow-dirty &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    errors=$((errors + 1))
fi

# Summary
echo "================================"
if [ $errors -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Ready to publish.${NC}"
    exit 0
else
    echo -e "${RED}✗ $errors test(s) failed. Please fix before publishing.${NC}"
    exit 1
fi
```

Make it executable:
```bash
chmod +x test-jsr.sh
./test-jsr.sh
```

## Quick Commands Reference

| Task | Command |
|------|---------|
| Lint | `npm run deno:lint` |
| Format | `npm run deno:fmt` |
| Check types | `npm run deno:check` |
| Dry run | `npm run jsr:publish:dry-run` |
| Publish | `npm run jsr:publish` |
| All checks | `./test-jsr.sh` |

## Next Steps

Once all tests pass:

1. ✅ Run `npm run jsr:publish:dry-run` one final time
2. ✅ Commit all changes
3. ✅ Tag the release: `git tag v0.1.28`
4. ✅ Push to GitHub: `git push && git push --tags`
5. ✅ Publish to npm: `npm publish`
6. ✅ Publish to JSR: `npm run jsr:publish`

## Resources

- [JSR Documentation](https://jsr.io/docs)
- [Deno Manual](https://deno.land/manual)
- [Publishing Packages to JSR](https://jsr.io/docs/publishing-packages)
