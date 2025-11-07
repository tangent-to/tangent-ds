# Package Registry & Runtime Ecosystem Guide

## JSR vs npm vs Browser: What You Need to Know

### The Key Question: Do You Need JSR?

**Short answer for tangent-ds**: Probably not critical right now, but nice to have for future-proofing.

### Why JSR Exists

JSR (JavaScript Registry) was created to solve specific problems:

1. **TypeScript-first**: No build step needed for TypeScript
2. **ESM-only**: No CommonJS legacy baggage
3. **Better documentation**: Auto-generated from TypeScript types
4. **Deno integration**: Native support in Deno runtime
5. **Module-level versioning**: Import specific versions per module

### JSR Technical Advantages

✅ **Has advantages when:**
- You write TypeScript and want to publish source (no compilation)
- You target Deno users
- You want automatic API documentation from types
- You need better ESM support

❌ **Doesn't help with:**
- Browser compatibility (separate concern)
- Performance
- Bun compatibility (Bun uses npm)
- Node.js adoption (npm is still dominant)

## Browser Compatibility: Separate from Registry

**Important**: JSR vs npm is about *where packages are stored*, not *where they run*.

### Your Current Setup (tangent-ds)

```json
"build:browser": "esbuild src/index.js --bundle --platform=browser ..."
```

This creates a browser bundle from your source. **This works the same whether you publish to npm or JSR.**

### How Users Consume Your Package

#### Option 1: npm Package (Current)
```javascript
// Install via npm
npm install @tangent.to/ds

// In Node.js/Bun
import { core, stats } from '@tangent.to/ds';

// In Browser (after bundling with webpack/vite/etc)
import { core, stats } from '@tangent.to/ds';

// Or via CDN (using your built dist/index.js)
<script type="module" src="https://unpkg.com/@tangent.to/ds/dist/index.js"></script>
```

#### Option 2: JSR Package (New)
```javascript
// In Deno
import { core, stats } from 'jsr:@tangent-to/ds';

// In Node.js (with compatibility layer)
import { core, stats } from '@tangent-to/ds'; // Still uses npm

// In Browser - JSR packages are NOT directly browser-compatible
// Users would need to bundle OR use an esm.sh proxy:
<script type="module">
  import { core } from 'https://esm.sh/jsr/@tangent-to/ds';
</script>
```

## Runtime Comparison

| Runtime | Primary Registry | TypeScript | Bundle Needed? |
|---------|------------------|------------|----------------|
| **Node.js** | npm | Needs compilation | Yes (for browser) |
| **Deno** | JSR or npm | Native support | Yes (for browser) |
| **Bun** | npm | Native support | Yes (for browser) |
| **Browser** | N/A (uses bundles) | Via bundler | Always |

## For tangent-ds Specifically

### Your Use Case Analysis

**What is tangent-ds?**
- Data science library for JavaScript
- Currently browser-focused (has esbuild browser bundle)
- Uses npm dependencies (ml-matrix, simple-statistics)
- Targets users who want client-side data analysis

**Who are your users?**
1. **Browser users** (via CDN or bundler)
2. **Node.js users** (server-side analysis)
3. **Observable users** (Observable notebooks)
4. **Potentially Deno users** (future)

### Should You Publish to JSR?

#### ✅ Reasons TO publish to JSR:
- **Future-proofing**: Deno ecosystem is growing
- **Better docs**: If you add TypeScript/JSDoc, JSR auto-generates docs
- **Modern tooling**: JSR is built for modern JavaScript
- **Zero-config TypeScript**: If you migrate to TypeScript later
- **Deno users**: Small but growing community

#### ❌ Reasons NOT to (or not to prioritize):
- **Limited browser benefit**: Doesn't make browser usage easier
- **npm still dominant**: 99% of your users will use npm
- **Dependencies on npm**: ml-matrix and simple-statistics are npm packages
- **Maintenance overhead**: Two registries to maintain
- **Bundling still needed**: Browser users still need bundled code

## Recommendation for tangent-ds

### Tier 1 Priority: npm
**Why**: This is where your users are.

```bash
npm publish
```

**Benefits**:
- Works with Node.js, Bun, and browser bundlers
- Largest ecosystem
- Your dependencies are already there
- CDN support (unpkg, jsdelivr, esm.sh)

### Tier 2 Priority: JSR (Optional)
**Why**: Nice to have for Deno users, minimal effort now that it's set up.

```bash
npm run jsr:publish
```

**Benefits**:
- Supports Deno users (small but growing)
- Future-proof if Deno adoption increases
- Better documentation if you add JSDoc/TypeScript
- Already configured, low maintenance

## Browser Usage Strategies

JSR doesn't change how browsers work. Here are your browser options:

### Strategy 1: Pre-built Bundle (Current Approach)
**What you're doing now**: Build with esbuild, publish to npm

```json
"build:browser": "esbuild src/index.js --bundle --platform=browser ..."
```

**User consumption**:
```html
<script type="module" src="https://unpkg.com/@tangent.to/ds/dist/index.js"></script>
```

✅ **Best for**: Direct browser usage, CDN distribution
✅ **Registry-agnostic**: Works from npm or JSR

### Strategy 2: Unbundled ESM (Let users bundle)
**Approach**: Publish source, let users bundle

**User consumption**:
```javascript
// In their app (using Vite/webpack/etc)
import { core } from '@tangent.to/ds';
// Their bundler handles it
```

✅ **Best for**: App developers who use bundlers
✅ **Registry-agnostic**: Works from npm or JSR

### Strategy 3: CDN with ESM proxy
**Approach**: Let CDNs handle bundling

**User consumption**:
```html
<script type="module">
  import { core } from 'https://esm.sh/@tangent.to/ds';
</script>
```

✅ **Best for**: Quick prototyping, notebooks
✅ **Works with**: npm packages (esm.sh can also serve JSR)

## The Bottom Line

### For tangent-ds:

1. **npm is essential** ✅
   - This is your primary distribution channel
   - Supports all runtimes (Node.js, Bun, browser via bundlers)
   - Required for Observable notebooks and most users

2. **JSR is nice-to-have** ⭐
   - Good for Deno users
   - Future-proofing
   - Better docs (if you add types)
   - Already set up, so why not?

3. **Browser compatibility is separate** 🌐
   - Determined by your code and build process
   - NOT determined by registry choice
   - Your esbuild setup handles this

### Recommended Publishing Strategy:

```bash
# Primary (always do this)
npm publish

# Secondary (if you have time)
npm run jsr:publish
```

### Migration Plan Reconsideration

You mentioned "migrating to JSR in the long run" - **reconsider this**:

- **Don't migrate away from npm** - it's still the dominant registry
- **Do publish to both** - dual publishing is cheap once set up
- **Think of JSR as supplementary** - not a replacement

### If You Had to Choose One:

**Choose npm** - hands down.

JSR is great for:
- Deno-first libraries
- TypeScript-heavy packages
- Internal tooling for Deno users

npm is better for:
- Browser-focused libraries (like yours)
- Maximum compatibility
- Largest user base

## Summary Table

| Aspect | npm | JSR |
|--------|-----|-----|
| Browser CDN support | ✅ Excellent (unpkg, jsdelivr) | ⚠️ Via esm.sh proxy |
| Node.js support | ✅ Native | ⚠️ Requires jsr npm package |
| Deno support | ✅ Via npm: specifier | ✅ Native |
| Bun support | ✅ Native | ❌ Uses npm anyway |
| Observable notebooks | ✅ Works great | ❌ Doesn't work |
| TypeScript support | ⚠️ Need to publish .d.ts | ✅ Native |
| User base | ✅✅✅ Massive | ⭐ Growing |
| Documentation | ⚠️ Manual | ✅ Auto-generated |

## Conclusion

**For tangent-ds, I recommend:**

1. **Primary**: Keep publishing to npm (this is your main channel)
2. **Secondary**: Also publish to JSR (already set up, helps Deno users)
3. **Browser**: Continue using esbuild to create browser bundles (registry-independent)

**Don't think of it as "migrating to JSR"** - think of it as "supporting both ecosystems."

The dual-publishing setup I created gives you the best of both worlds with minimal overhead.
