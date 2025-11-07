#!/usr/bin/env node

/**
 * Synchronize version across package.json, jsr.json, and deno.json
 *
 * Usage:
 *   node scripts/sync-version.js          # Show current versions
 *   node scripts/sync-version.js 0.1.29   # Update all to 0.1.29
 */

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

const configFiles = [
  'package.json',
  'jsr.json',
  'deno.json'
];

function readVersion(file) {
  const path = join(rootDir, file);
  const content = readFileSync(path, 'utf8');
  const json = JSON.parse(content);
  return json.version;
}

function writeVersion(file, version) {
  const path = join(rootDir, file);
  const content = readFileSync(path, 'utf8');
  const json = JSON.parse(content);
  json.version = version;
  writeFileSync(path, JSON.stringify(json, null, 2) + '\n');
}

function validateVersion(version) {
  // Basic semver validation
  const semverRegex = /^\d+\.\d+\.\d+(-[\w.]+)?$/;
  if (!semverRegex.test(version)) {
    console.error(`❌ Invalid version format: ${version}`);
    console.error('Expected format: X.Y.Z or X.Y.Z-prerelease');
    process.exit(1);
  }
}

function main() {
  const newVersion = process.argv[2];

  if (!newVersion) {
    // Show current versions
    console.log('Current versions:');
    console.log('─'.repeat(40));
    configFiles.forEach(file => {
      const version = readVersion(file);
      console.log(`${file.padEnd(20)} ${version}`);
    });
    console.log('─'.repeat(40));
    console.log('\nUsage: node scripts/sync-version.js <version>');
    console.log('Example: node scripts/sync-version.js 0.1.29');
    return;
  }

  validateVersion(newVersion);

  console.log(`Updating all versions to ${newVersion}...`);
  console.log('─'.repeat(40));

  configFiles.forEach(file => {
    const oldVersion = readVersion(file);
    writeVersion(file, newVersion);
    console.log(`✓ ${file.padEnd(20)} ${oldVersion} → ${newVersion}`);
  });

  console.log('─'.repeat(40));
  console.log('✓ All versions synchronized!');
  console.log('\nNext steps:');
  console.log('1. Review changes: git diff');
  console.log('2. Run tests: npm run test:run');
  console.log('3. Commit: git commit -am "chore: bump version to ' + newVersion + '"');
  console.log('4. Tag: git tag v' + newVersion);
  console.log('5. Push: git push && git push --tags');
}

main();
