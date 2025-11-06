/**
 * R-style Formula Parser
 *
 * Supports:
 * - Basic formulas: 'y ~ x1 + x2'
 * - Interactions: 'y ~ x1 * x2' (expands to x1 + x2 + x1:x2)
 * - Transformations: 'y ~ log(x1) + sqrt(x2)'
 * - I() for inline expressions: 'y ~ I(x^2)'
 * - Polynomials: 'y ~ poly(x, 3)'
 * - Mixed effects: 'y ~ x1 + (1 | group)' or 'y ~ x1 + (1 + time | subject)'
 */

/**
 * Parse an R-style formula string
 * @param {string} formula - Formula string like 'y ~ x1 + x2'
 * @param {Array<Object>|Object} data - Data array or table
 * @returns {Object} Parsed formula specification
 */
export function parseFormula(formula, data = null) {
  // Split on ~ to get response and predictors
  const parts = formula.split('~').map(s => s.trim());

  if (parts.length !== 2) {
    throw new Error(`Invalid formula: ${formula}. Expected format 'y ~ x1 + x2'`);
  }

  const [responsePart, predictorsPart] = parts;

  // Parse response (left side)
  const response = parseResponse(responsePart);

  // Parse predictors (right side), including random effects
  const { fixed, random } = parsePredictors(predictorsPart);

  return {
    response,
    fixed,
    random,
    original: formula
  };
}

/**
 * Parse response variable (left side of ~)
 */
function parseResponse(part) {
  const trimmed = part.trim();

  // Check for transformation
  if (trimmed.includes('(')) {
    const match = trimmed.match(/^(\w+)\((.*)\)$/);
    if (match) {
      return {
        variable: match[2],
        transform: match[1]
      };
    }
  }

  return {
    variable: trimmed,
    transform: null
  };
}

/**
 * Parse predictor side (right side of ~)
 */
function parsePredictors(part) {
  // Extract random effects first (anything in parentheses with |)
  const randomEffects = [];
  let fixedPart = part;

  // Match random effects: (terms | grouping)
  const randomRegex = /\(([^|]+)\|([^)]+)\)/g;
  let match;

  while ((match = randomRegex.exec(part)) !== null) {
    const terms = match[1].trim();
    const grouping = match[2].trim();
    randomEffects.push({ terms, grouping });

    // Remove this random effect from the fixed part
    fixedPart = fixedPart.replace(match[0], '');
  }

  // Parse fixed effects
  const fixedTerms = parseFixedTerms(fixedPart);

  // Parse random effects structure
  const random = randomEffects.length > 0
    ? parseRandomEffects(randomEffects)
    : null;

  return {
    fixed: fixedTerms,
    random
  };
}

/**
 * Parse fixed effects terms
 */
function parseFixedTerms(part) {
  const terms = [];
  const tokens = tokenize(part);

  let i = 0;
  while (i < tokens.length) {
    const token = tokens[i];

    if (token === '+' || token === '-') {
      i++;
      continue;
    }

    if (token === '*') {
      // Interaction: expand a * b to a + b + a:b
      // This requires looking back at the previous term
      i++;
      continue;
    }

    if (token === ':') {
      // Explicit interaction
      i++;
      continue;
    }

    // Parse term
    const term = parseTerm(token);
    terms.push(term);

    i++;
  }

  // Expand interactions
  const expanded = expandInteractions(tokens, terms);

  return expanded;
}

/**
 * Tokenize formula string
 */
function tokenize(str) {
  const tokens = [];
  let current = '';
  let parenDepth = 0;

  for (let i = 0; i < str.length; i++) {
    const char = str[i];

    if (char === '(') {
      parenDepth++;
      current += char;
    } else if (char === ')') {
      parenDepth--;
      current += char;
    } else if ((char === '+' || char === '-' || char === '*' || char === ':') && parenDepth === 0) {
      if (current.trim()) {
        tokens.push(current.trim());
      }
      tokens.push(char);
      current = '';
    } else if (char === ' ' && parenDepth === 0) {
      // Skip whitespace outside parentheses
      continue;
    } else {
      current += char;
    }
  }

  if (current.trim()) {
    tokens.push(current.trim());
  }

  return tokens;
}

/**
 * Parse a single term
 */
function parseTerm(token) {
  // Check for function calls
  const funcMatch = token.match(/^(\w+)\((.*)\)$/);
  if (funcMatch) {
    const func = funcMatch[1];
    const args = funcMatch[2];

    // Handle special functions
    if (func === 'I') {
      // Inline expression: I(x^2)
      return {
        type: 'expression',
        expression: args,
        variables: extractVariables(args)
      };
    } else if (func === 'poly') {
      // Polynomial: poly(x, 3)
      const [variable, degree] = args.split(',').map(s => s.trim());
      return {
        type: 'polynomial',
        variable,
        degree: parseInt(degree)
      };
    } else {
      // Transform: log(x), sqrt(x), etc.
      return {
        type: 'transform',
        transform: func,
        variable: args.trim()
      };
    }
  }

  // Simple variable
  return {
    type: 'variable',
    name: token
  };
}

/**
 * Extract variable names from an expression
 */
function extractVariables(expr) {
  // Simple extraction - just find word characters
  const matches = expr.match(/\b[a-zA-Z_]\w*\b/g);
  return matches ? [...new Set(matches)] : [];
}

/**
 * Expand interactions (a * b becomes a + b + a:b)
 */
function expandInteractions(tokens, terms) {
  const expanded = [...terms];

  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === '*') {
      // Find the terms before and after *
      let beforeIdx = i - 1;
      while (beforeIdx >= 0 && (tokens[beforeIdx] === '+' || tokens[beforeIdx] === '-')) {
        beforeIdx--;
      }

      let afterIdx = i + 1;
      while (afterIdx < tokens.length && (tokens[afterIdx] === '+' || tokens[afterIdx] === '-')) {
        afterIdx++;
      }

      if (beforeIdx >= 0 && afterIdx < tokens.length) {
        const term1 = parseTerm(tokens[beforeIdx]);
        const term2 = parseTerm(tokens[afterIdx]);

        // Add interaction term
        expanded.push({
          type: 'interaction',
          terms: [term1, term2]
        });
      }
    }
  }

  return expanded;
}

/**
 * Parse random effects structure
 */
function parseRandomEffects(randomEffects) {
  const structure = {
    intercept: null,
    slopes: {}
  };

  for (const { terms, grouping } of randomEffects) {
    const termsList = terms.split('+').map(t => t.trim());

    for (const term of termsList) {
      if (term === '1') {
        // Random intercept
        structure.intercept = grouping;
      } else if (term === '0') {
        // Suppress intercept (rare)
        continue;
      } else {
        // Random slope
        if (!structure.slopes[term]) {
          structure.slopes[term] = grouping;
        }
      }
    }
  }

  return structure;
}

/**
 * Apply formula to data to extract design matrix and response
 * @param {string} formula - Formula string
 * @param {Array<Object>} data - Data array
 * @param {Object} options - Additional options
 * @returns {Object} {X, y, groups, columnNames}
 */
export function applyFormula(formula, data, options = {}) {
  const parsed = parseFormula(formula, data);

  // Extract response variable
  const y = extractResponse(parsed.response, data);

  // Build design matrix from fixed effects
  const { X, columnNames } = buildDesignMatrix(parsed.fixed, data, options);

  // Extract grouping variables for random effects
  let groups = null;
  let randomEffectsData = null;

  if (parsed.random) {
    randomEffectsData = extractRandomEffects(parsed.random, data, X);

    // For backward compatibility, also provide simple groups array
    if (parsed.random.intercept) {
      groups = data.map(row => row[parsed.random.intercept]);
    }
  }

  return {
    X,
    y,
    groups,
    randomEffects: randomEffectsData,
    columnNames,
    parsed
  };
}

/**
 * Extract response variable with optional transformation
 */
function extractResponse(response, data) {
  const values = data.map(row => row[response.variable]);

  if (response.transform) {
    return applyTransform(values, response.transform);
  }

  return values;
}

/**
 * Build design matrix from fixed effects terms
 */
function buildDesignMatrix(terms, data, options = {}) {
  const { intercept = true } = options;

  const allColumns = [];
  const columnNames = [];

  if (intercept) {
    columnNames.push('(Intercept)');
    allColumns.push(Array(data.length).fill(1));
  }

  // Process each term and collect columns
  for (const term of terms) {
    const { columns, names } = processTerm(term, data);

    allColumns.push(...columns);
    columnNames.push(...names);
  }

  // Build design matrix: transpose columns to rows
  const X = data.map((_, i) =>
    allColumns.map(col => col[i])
  );

  return { X, columnNames };
}

/**
 * Process a single term to extract column(s)
 */
function processTerm(term, data) {
  if (term.type === 'variable') {
    return {
      columns: [data.map(row => row[term.name])],
      names: [term.name]
    };
  }

  if (term.type === 'transform') {
    const values = data.map(row => row[term.variable]);
    const transformed = applyTransform(values, term.transform);
    return {
      columns: [transformed],
      names: [`${term.transform}(${term.variable})`]
    };
  }

  if (term.type === 'expression') {
    // Evaluate expression for each row
    const values = data.map(row => evaluateExpression(term.expression, row));
    return {
      columns: [values],
      names: [`I(${term.expression})`]
    };
  }

  if (term.type === 'polynomial') {
    // Generate polynomial terms
    const baseValues = data.map(row => row[term.variable]);
    const columns = [];
    const names = [];

    for (let d = 1; d <= term.degree; d++) {
      columns.push(baseValues.map(v => Math.pow(v, d)));
      names.push(d === 1 ? term.variable : `${term.variable}^${d}`);
    }

    return { columns, names };
  }

  if (term.type === 'interaction') {
    // Compute interaction (element-wise product)
    const term1Data = processTerm(term.terms[0], data);
    const term2Data = processTerm(term.terms[1], data);

    const interaction = term1Data.columns[0].map((v1, i) =>
      v1 * term2Data.columns[0][i]
    );

    return {
      columns: [interaction],
      names: [`${term1Data.names[0]}:${term2Data.names[0]}`]
    };
  }

  return { columns: [[]], names: [] };
}

/**
 * Apply a transformation function
 */
function applyTransform(values, transform) {
  const transformFuncs = {
    log: Math.log,
    log10: Math.log10,
    log2: Math.log2,
    sqrt: Math.sqrt,
    exp: Math.exp,
    abs: Math.abs,
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan
  };

  const func = transformFuncs[transform];
  if (!func) {
    throw new Error(`Unknown transformation: ${transform}`);
  }

  return values.map(func);
}

/**
 * Evaluate an expression for a given row
 */
function evaluateExpression(expr, row) {
  // Replace variables with their values
  let evalExpr = expr;

  for (const [key, value] of Object.entries(row)) {
    // Use word boundaries to avoid partial replacements
    const regex = new RegExp(`\\b${key}\\b`, 'g');
    evalExpr = evalExpr.replace(regex, value);
  }

  // Replace ^ with **
  evalExpr = evalExpr.replace(/\^/g, '**');

  // Evaluate safely (note: eval is generally unsafe, but we're in a controlled context)
  try {
    return Function('"use strict"; return (' + evalExpr + ')')();
  } catch (e) {
    throw new Error(`Failed to evaluate expression: ${expr}`);
  }
}

/**
 * Extract random effects grouping variables
 */
function extractRandomEffects(random, data, X) {
  const result = {};

  if (random.intercept) {
    result.intercept = data.map(row => row[random.intercept]);
  }

  if (Object.keys(random.slopes).length > 0) {
    result.slopes = {};

    for (const [variable, grouping] of Object.entries(random.slopes)) {
      result.slopes[variable] = {
        groups: data.map(row => row[grouping]),
        values: data.map(row => row[variable])
      };
    }
  }

  return result;
}
