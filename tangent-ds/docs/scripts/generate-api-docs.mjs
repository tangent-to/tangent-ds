import {promises as fs} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiDocsDir = path.resolve(__dirname, '..', 'docs', 'api');
const srcDir = path.resolve(__dirname, '..', '..', 'src');

// Clean the API docs directory
async function cleanApiDocs() {
  try {
    await fs.rm(apiDocsDir, {recursive: true, force: true});
  } catch (error) {
    // Ignore if directory doesn't exist
  }
  await fs.mkdir(apiDocsDir, {recursive: true});
}

// Run jsdoc2md command
function runJsdoc2md(files, outputFile) {
  return new Promise((resolve, reject) => {
    const jsdoc2md = spawn('npx', [
      'jsdoc2md',
      '--files',
      ...files,
      '--configure',
      './jsdoc.json',
    ]);

    let stdout = '';
    let stderr = '';

    jsdoc2md.stdout.on('data', data => {
      stdout += data.toString();
    });

    jsdoc2md.stderr.on('data', data => {
      stderr += data.toString();
    });

    jsdoc2md.on('close', code => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`jsdoc2md failed: ${stderr}`));
      }
    });

    jsdoc2md.on('error', reject);
  });
}

// Get all JS files from src directory
async function getSourceFiles(dir) {
  const entries = await fs.readdir(dir, {withFileTypes: true});
  const files = await Promise.all(
    entries.map(async entry => {
      const resolved = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        return getSourceFiles(resolved);
      }
      return entry.isFile() && entry.name.endsWith('.js') ? resolved : [];
    }),
  );
  return files.flat();
}

// Generate API docs
async function generateApiDocs() {
  console.log('[INFO] Cleaning API docs directory...');
  await cleanApiDocs();

  console.log('[INFO] Finding source files...');
  const sourceFiles = await getSourceFiles(srcDir);
  console.log(`[INFO] Found ${sourceFiles.length} source files`);

  console.log('[INFO] Generating API documentation...');
  let markdown = await runJsdoc2md(sourceFiles);

  // Escape JSX-like syntax for MDX compatibility
  console.log('[INFO] Post-processing markdown for MDX compatibility...');

  // Replace curly braces with HTML entities in text (but not in code blocks)
  const lines = markdown.split('\n');
  let inCodeBlock = false;
  const processedLines = lines.map(line => {
    // Track code blocks
    if (line.trim().startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      return line;
    }

    // Skip lines inside code blocks
    if (inCodeBlock) {
      return line;
    }

    // Escape curly braces in regular text (not in backticks or HTML tags)
    const parts = line.split(/(`[^`]*`)/);
    const escaped = parts.map((part, i) => {
      // Keep inline code (odd indices) unchanged
      if (i % 2 === 1) return part;
      // Replace curly braces with HTML entities, but preserve HTML tags
      // Don't escape < and > as they're needed for HTML tags in markdown
      return part
        .replace(/{/g, '&#123;')
        .replace(/}/g, '&#125;');
    });
    return escaped.join('');
  });

  markdown = processedLines.join('\n');

  // Write the main API file
  const apiFile = path.join(apiDocsDir, 'index.md');
  const frontMatter = `---
id: api
title: API Reference
sidebar_label: API
---

`;

  await fs.writeFile(apiFile, frontMatter + markdown);
  console.log(`[INFO] API documentation generated at ${apiFile}`);
}

await generateApiDocs();
