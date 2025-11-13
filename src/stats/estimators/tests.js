import { Estimator } from '../../core/estimators/estimator.js';
import { normalize } from '../../core/table.js';
import {
  oneSampleTTest as oneSampleTTestFn,
  twoSampleTTest as twoSampleTTestFn,
  chiSquareTest as chiSquareTestFn,
  oneWayAnova as oneWayAnovaFn,
  tukeyHSD as tukeyHSDFn
} from '../tests.js';

function ensureNumeric(value, columnName) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`Column ${columnName} contains non-numeric value: ${value}`);
  }
  return value;
}

function extractColumn(data, column, omitMissing = true) {
  if (!data) {
    throw new Error('Data is required for declarative usage.');
  }

  const rows = normalize(data);
  const filtered = omitMissing
    ? rows.filter((row) => row[column] != null && !Number.isNaN(row[column]))
    : rows;

  return filtered.map((row) => ensureNumeric(row[column], column));
}

class StatisticalTest extends Estimator {
  constructor(params = {}) {
    super(params);
    this.result = null;
  }

  summary() {
    if (!this.fitted || !this.result) {
      throw new Error('Statistical test has not been fitted. Call fit() first.');
    }
    return this.result;
  }

  toJSON() {
    return {
      __class__: this.constructor.name,
      params: this.getParams(),
      fitted: !!this.fitted,
      result: this.result
    };
  }

  static fromJSON(obj = {}) {
    const inst = new this(obj.params || {});
    if (obj.result) {
      inst.result = obj.result;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

const ONE_SAMPLE_DEFAULTS = {
  mu: 0,
  alternative: 'two-sided',
  omit_missing: true
};

export class OneSampleTTest extends StatisticalTest {
  fit(sample, opts = {}) {
    let values = sample;
    let options = { ...ONE_SAMPLE_DEFAULTS, ...this.params, ...opts };

    if (
      sample &&
      typeof sample === 'object' &&
      !Array.isArray(sample)
    ) {
      if (!sample.data || !sample.column) {
        throw new Error('OneSampleTTest declarative usage requires `data` and `column`.');
      }
      options = { ...options, ...sample };
      values = extractColumn(sample.data, sample.column, options.omit_missing);
    }

    if (!Array.isArray(values)) {
      throw new Error('OneSampleTTest.fit expects an array of values or a declarative object.');
    }

    this.result = oneSampleTTestFn(values, {
      mu: options.mu,
      alternative: options.alternative
    });
    this.fitted = true;
    return this;
  }
}

const TWO_SAMPLE_DEFAULTS = {
  alternative: 'two-sided',
  omit_missing: true
};

function splitTwoGroups(rows, groupColumn, valueColumn, omitMissing) {
  const groups = new Map();
  for (const row of rows) {
    const groupKey = row[groupColumn];
    const value = row[valueColumn];
    if (omitMissing && (groupKey == null || value == null || Number.isNaN(value))) {
      continue;
    }
    ensureNumeric(value, valueColumn);
    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey).push(value);
  }

  if (groups.size !== 2) {
    throw new Error('TwoSampleTTest declarative usage requires exactly two groups.');
  }

  return Array.from(groups.values());
}

export class TwoSampleTTest extends StatisticalTest {
  fit(sample1, sample2 = null, opts = {}) {
    let values1 = sample1;
    let values2 = sample2;
    let options = { ...TWO_SAMPLE_DEFAULTS, ...this.params, ...opts };

    if (
      sample1 &&
      typeof sample1 === 'object' &&
      !Array.isArray(sample1)
    ) {
      const cfg = sample1;
      if (!cfg.data || !cfg.groups || !cfg.value) {
        throw new Error('TwoSampleTTest declarative usage requires `data`, `groups`, and `value`.');
      }
      options = { ...options, ...cfg };
      const rows = normalize(cfg.data);
      const [groupA, groupB] = splitTwoGroups(
        rows,
        cfg.groups,
        cfg.value,
        options.omit_missing
      );
      values1 = groupA;
      values2 = groupB;
    }

    if (!Array.isArray(values1) || !Array.isArray(values2)) {
      throw new Error('TwoSampleTTest.fit expects two numeric arrays or a declarative object.');
    }

    this.result = twoSampleTTestFn(values1, values2, {
      alternative: options.alternative
    });
    this.fitted = true;
    return this;
  }
}

const ANOVA_DEFAULTS = {
  omit_missing: true
};

function splitGroups(rows, groupColumn, valueColumn, omitMissing) {
  const groups = new Map();
  for (const row of rows) {
    const groupKey = row[groupColumn];
    const value = row[valueColumn];
    if (omitMissing && (groupKey == null || value == null || Number.isNaN(value))) {
      continue;
    }
    ensureNumeric(value, valueColumn);
    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey).push(value);
  }

  if (groups.size < 2) {
    throw new Error('OneWayAnova declarative usage requires at least two groups.');
  }

  return Array.from(groups.values());
}

export class OneWayAnova extends StatisticalTest {
  fit(groups, opts = {}) {
    let samples = groups;
    let options = { ...ANOVA_DEFAULTS, ...this.params, ...opts };

    if (
      groups &&
      typeof groups === 'object' &&
      !Array.isArray(groups)
    ) {
      if (!groups.data || !groups.groups || !groups.value) {
        throw new Error('OneWayAnova declarative usage requires `data`, `groups`, and `value`.');
      }
      options = { ...options, ...groups };
      const rows = normalize(groups.data);
      samples = splitGroups(rows, groups.groups, groups.value, options.omit_missing);
    }

    if (!Array.isArray(samples) || samples.length === 0 || !Array.isArray(samples[0])) {
      throw new Error('OneWayAnova.fit expects an array of numeric arrays or a declarative object.');
    }

    this.result = oneWayAnovaFn(samples);
    this.fitted = true;
    return this;
  }
}

const CHI_SQUARE_DEFAULTS = {
  omit_missing: true
};

export class ChiSquareTest extends StatisticalTest {
  fit(observed, expected = null, opts = {}) {
    let obs = observed;
    let exp = expected;
    let options = { ...CHI_SQUARE_DEFAULTS, ...this.params, ...opts };

    if (
      observed &&
      typeof observed === 'object' &&
      !Array.isArray(observed)
    ) {
      if (!observed.data || !observed.observed || !observed.expected) {
        throw new Error('ChiSquareTest declarative usage requires `data`, `observed`, and `expected`.');
      }
      options = { ...options, ...observed };
      const rows = normalize(observed.data);
      const omit = options.omit_missing;

      obs = [];
      exp = [];

      for (const row of rows) {
        const obsVal = row[observed.observed];
        const expVal = row[observed.expected];
        if (omit && (
          obsVal == null || Number.isNaN(obsVal) ||
          expVal == null || Number.isNaN(expVal)
        )) {
          continue;
        }
        obs.push(ensureNumeric(obsVal, observed.observed));
        exp.push(ensureNumeric(expVal, observed.expected));
      }
    }

    if (!Array.isArray(obs) || !Array.isArray(exp)) {
      throw new Error('ChiSquareTest.fit expects observed/expected arrays or a declarative object.');
    }

    this.result = chiSquareTestFn(obs, exp);
    this.fitted = true;
    return this;
  }
}

const TUKEY_HSD_DEFAULTS = {
  alpha: 0.05,
  omit_missing: true
};

export class TukeyHSD extends StatisticalTest {
  fit(groups, opts = {}) {
    let samples = groups;
    let options = { ...TUKEY_HSD_DEFAULTS, ...this.params, ...opts };

    if (
      groups &&
      typeof groups === 'object' &&
      !Array.isArray(groups)
    ) {
      if (!groups.data || !groups.groups || !groups.value) {
        throw new Error('TukeyHSD declarative usage requires `data`, `groups`, and `value`.');
      }
      options = { ...options, ...groups };
      const rows = normalize(groups.data);
      samples = splitGroups(rows, groups.groups, groups.value, options.omit_missing);
    }

    if (!Array.isArray(samples) || samples.length === 0 || !Array.isArray(samples[0])) {
      throw new Error('TukeyHSD.fit expects an array of numeric arrays or a declarative object.');
    }

    this.result = tukeyHSDFn(samples, {
      alpha: options.alpha,
      anovaResult: options.anovaResult
    });
    this.fitted = true;
    return this;
  }
}

// Attach functional helpers to the classes to keep a single touchpoint.
Object.assign(OneSampleTTest, { compute: oneSampleTTestFn });
Object.assign(TwoSampleTTest, { compute: twoSampleTTestFn });
Object.assign(OneWayAnova, { compute: oneWayAnovaFn });
Object.assign(ChiSquareTest, { compute: chiSquareTestFn });
Object.assign(TukeyHSD, { compute: tukeyHSDFn });

export default {
  OneSampleTTest,
  TwoSampleTTest,
  OneWayAnova,
  ChiSquareTest,
  TukeyHSD
};
