import { describe, it, expect } from 'vitest';
// CCA temporarily disabled due to Observable performance issues
// import { cca, CCA } from '../src/mva/index.js';

function generateData() {
  const X = [];
  const Y = [];

  for (let i = 0; i < 12; i++) {
    const x1 = i;
    const x2 = Math.sin(i / 2) + i * 0.1;
    const x3 = Math.cos(i / 3);

    const y1 = 0.6 * x1 + 0.2 * x2 + 0.1 * x3;
    const y2 = -0.3 * x1 + 0.8 * x2 + 0.05 * x3;

    X.push([x1, x2, x3]);
    Y.push([y1, y2]);
  }

  return { X, Y };
}

describe.skip('CCA - functional API', () => {
  it('should compute canonical correlations for numeric matrices', () => {
    const { X, Y } = generateData();
    const model = cca.fit(X, Y);

    expect(model.type).toBe('cca');
    expect(model.correlations.length).toBe(2);
    expect(model.correlations[0]).toBeGreaterThan(0.99);
    expect(model.xScores.length).toBe(X.length);
    expect(model.yScores.length).toBe(Y.length);
  });

  it('should support declarative input with table data', () => {
    const rows = [];
    for (let i = 0; i < 10; i++) {
      const x1 = i * 0.5;
      const x2 = i - 1;
      const y1 = x1 * 0.8 + x2 * 0.2;
      const y2 = x1 * -0.2 + x2 * 0.9;
      rows.push({
        x1,
        x2,
        y1,
        y2
      });
    }

    const model = cca.fit({
      data: rows,
      X: ['x1', 'x2'],
      Y: ['y1', 'y2']
    });

    expect(model.correlations[0]).toBeGreaterThan(0.99);
    expect(model.columnsX).toEqual(['x1', 'x2']);
    expect(model.columnsY).toEqual(['y1', 'y2']);
  });

  it('transform should reproduce fitted scores on training data', () => {
    const { X, Y } = generateData();
    const model = cca.fit(X, Y);

    const transformed = cca.transform(model, X, Y);

    expect(transformed.xScores[0].cca1).toBeCloseTo(model.xScores[0].cca1, 6);
    expect(transformed.yScores[5].cca1).toBeCloseTo(model.yScores[5].cca1, 6);
  });
});

describe.skip('CCA - estimator class', () => {
  it('should fit and provide summary', () => {
    const { X, Y } = generateData();
    const estimator = new CCA();
    estimator.fit(X, Y);

    const summary = estimator.summary();
    expect(summary.nComponents).toBe(2);
    expect(summary.correlations[0]).toBeGreaterThan(0.99);
  });

  it('should transform X and Y using estimator methods', () => {
    const { X, Y } = generateData();
    const estimator = new CCA();
    estimator.fit(X, Y);

    const xScores = estimator.transformX(X);
    const yScores = estimator.transformY(Y);

    expect(xScores.length).toBe(X.length);
    expect(yScores.length).toBe(Y.length);
    expect(xScores[0].cca1).toBeCloseTo(estimator.model.xScores[0].cca1, 6);
  });
});
