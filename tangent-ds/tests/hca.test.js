import { describe, it, expect, vi } from 'vitest';
import { fit, cut, cutHeight } from '../src/mva/hca.js';
import { HCA } from '../src/mva/index.js';
import { plotHCA } from '../src/plot/plotHCA.js';
import { createD3DendrogramRenderer } from '../src/plot/renderers/d3Dendrogram.js';

function createMockDocument() {
  class MockElement {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.attributes = new Map();
      this.children = [];
      this.dataset = {};
      this.style = {};
      this.textContent = '';
      this.parentNode = null;
    }

    setAttribute(name, value) {
      this.attributes.set(name, String(value));
    }

    getAttribute(name) {
      const value = this.attributes.get(name);
      return value === undefined ? null : value;
    }

    appendChild(child) {
      child.parentNode = this;
      this.children.push(child);
      return child;
    }

    querySelectorAll(tag) {
      const matches = [];
      const target = tag.toUpperCase();

      const visit = (node) => {
        node.children.forEach((child) => {
          if (child.tagName === target) {
            matches.push(child);
          }
          visit(child);
        });
      };

      visit(this);
      return matches;
    }
  }

  return {
    createElementNS(_namespace, tag) {
      return new MockElement(tag);
    }
  };
}

const originalDocument = globalThis.document;

beforeAll(() => {
  if (typeof document === 'undefined') {
    globalThis.document = createMockDocument();
  }
});

afterAll(() => {
  if (originalDocument === undefined) {
    delete globalThis.document;
  } else {
    globalThis.document = originalDocument;
  }
});

describe('HCA - Hierarchical Clustering', () => {
  describe('fit', () => {
    it('should perform hierarchical clustering', () => {
      const X = [
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11]
      ];
      
      const model = fit(X, { linkage: 'average' });
      
      expect(model.dendrogram.length).toBe(3); // n-1 merges
      expect(model.linkage).toBe('average');
      expect(model.n).toBe(4);
    });

    it('should work with single linkage', () => {
      const X = [[1], [2], [10], [11]];
      const model = fit(X, { linkage: 'single' });
      
      expect(model.dendrogram.length).toBe(3);
      expect(model.linkage).toBe('single');
    });

    it('should work with complete linkage', () => {
      const X = [[1], [2], [10]];
      const model = fit(X, { linkage: 'complete' });
      
      expect(model.dendrogram.length).toBe(2);
      expect(model.linkage).toBe('complete');
    });

    it('should work with ward linkage', () => {
      const X = [
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11]
      ];
      const model = fit(X, { linkage: 'ward' });

      expect(model.dendrogram.length).toBe(3);
      expect(model.linkage).toBe('ward');
      expect(model.dendrogram[0].distance).toBeCloseTo(0.5, 5);
      expect(model.dendrogram[1].distance).toBeGreaterThanOrEqual(model.dendrogram[0].distance);
    });

    it('should have increasing distances in dendrogram', () => {
      const X = [[0], [1], [2], [10]];
      const model = fit(X);
      
      // Distances should generally increase (not strict for all linkages)
      expect(model.dendrogram.every(m => m.distance >= 0)).toBe(true);
    });

    it('should throw error for insufficient samples', () => {
      const X = [[1]];
      expect(() => fit(X)).toThrow();
    });
  });

  describe('cut', () => {
    it('should cut dendrogram into k clusters', () => {
      const X = [
        [0, 0], [0, 1],    // Cluster 1
        [10, 10], [10, 11] // Cluster 2
      ];
      
      const model = fit(X);
      const labels = cut(model, 2);
      
      expect(labels.length).toBe(4);
      expect(new Set(labels).size).toBe(2);
      
      // Points 0,1 should be in same cluster
      expect(labels[0]).toBe(labels[1]);
      // Points 2,3 should be in same cluster
      expect(labels[2]).toBe(labels[3]);
      // But different from first cluster
      expect(labels[0]).not.toBe(labels[2]);
    });

    it('should handle k=n (all separate)', () => {
      const X = [[1], [2], [3]];
      const model = fit(X);
      const labels = cut(model, 3);
      
      expect(labels.length).toBe(3);
      expect(new Set(labels).size).toBe(3);
    });

    it('should handle k=1 (all together)', () => {
      const X = [[1], [2], [3]];
      const model = fit(X);
      const labels = cut(model, 1);
      
      expect(labels.length).toBe(3);
      expect(new Set(labels).size).toBe(1);
    });

    it('should throw error for invalid k', () => {
      const X = [[1], [2]];
      const model = fit(X);
      
      expect(() => cut(model, 0)).toThrow();
      expect(() => cut(model, 10)).toThrow();
    });
  });

  describe('cutHeight', () => {
    it('should cut dendrogram at specified height', () => {
      const X = [[0], [1], [10], [11]];
      const model = fit(X);
      
      // Cut at very low height - should get many clusters
      const labels1 = cutHeight(model, 0.5);
      expect(new Set(labels1).size).toBeGreaterThan(2);
      
      // Cut at high height - should get fewer clusters
      const labels2 = cutHeight(model, 10);
      expect(new Set(labels2).size).toBeLessThanOrEqual(2);
    });

    it('should produce valid cluster labels', () => {
      const X = [[1], [2], [3]];
      const model = fit(X);
      const labels = cutHeight(model, 1);
      
      expect(labels.length).toBe(3);
      expect(labels.every(l => l >= 0)).toBe(true);
    });
  });
});

describe('HCA - class API', () => {
  it('should fit using class wrapper and cut clusters', () => {
    const X = [
      [0, 0],
      [0, 1],
      [10, 10],
      [10, 11]
    ];

    const estimator = new HCA({ linkage: 'average' });
    estimator.fit(X);
    const labels = estimator.cut(2);

    expect(labels.length).toBe(4);
    expect(new Set(labels).size).toBe(2);
  });

  it('should provide summary information', () => {
    const X = [[1], [2], [3]];
    const estimator = new HCA();
    estimator.fit(X);
    const summary = estimator.summary();

    expect(summary.linkage).toBeDefined();
    expect(summary.merges).toBeGreaterThan(0);
  });

  it('should support ward linkage through estimator class', () => {
    const X = [
      [0, 0],
      [0, 1],
      [10, 10],
      [10, 11]
    ];

    const estimator = new HCA({ linkage: 'ward' });
    estimator.fit(X);

    const summary = estimator.summary();
    expect(summary.linkage).toBe('ward');

    const labels = estimator.cut(2);
    expect(new Set(labels).size).toBe(2);
  });
});

describe('HCA - visualization helpers', () => {
  const sampleData = [
    [0, 0],
    [0, 1],
    [10, 10],
    [10, 11]
  ];

  it('plotHCA attaches show helper requiring renderer', () => {
    const model = fit(sampleData, { linkage: 'average' });
    const spec = plotHCA(model);

    expect(typeof spec.show).toBe('function');
    expect(() => spec.show()).toThrow(/dendrogram renderer/);

    const renderFn = vi.fn(() => 'rendered');
    const result = spec.show(renderFn, { width: 800, orientation: 'horizontal' });

    expect(result).toBe('rendered');
    expect(renderFn).toHaveBeenCalledTimes(1);

    const renderArg = renderFn.mock.calls[0][0];
    expect(renderArg.type).toBe('dendrogram');
    expect(renderArg.config.width).toBe(800);
    expect(renderArg.config.orientation).toBe('horizontal');
    expect(renderArg.data).toEqual(spec.data);
    expect(spec.config.width).toBe(640);
  });

  it('plotHCA show supports renderer objects', () => {
    const model = fit(sampleData);
    const spec = plotHCA(model);
    const renderer = {
      render: vi.fn(() => 'ok')
    };

    const result = spec.show(renderer);
    expect(result).toBe('ok');
    expect(renderer.render).toHaveBeenCalledTimes(1);
  });

  it('plotHCA show rejects unsupported renderer objects', () => {
    const model = fit(sampleData);
    const spec = plotHCA(model);

    expect(() => spec.show({})).toThrow(/Unsupported dendrogram renderer/);
  });

  it('createD3DendrogramRenderer renders SVG output', () => {
    const model = fit(sampleData, { linkage: 'ward' });
    const spec = plotHCA(model);
    const fakeD3 = {
      scaleLinear() {
        let domain = [0, 1];
        let range = [0, 1];
        const scale = (value) => {
          const denom = domain[1] - domain[0];
          if (denom === 0) return range[0];
          const t = (value - domain[0]) / denom;
          return range[0] + t * (range[1] - range[0]);
        };
        scale.domain = (values) => {
          domain = values.slice();
          return scale;
        };
        scale.range = (values) => {
          range = values.slice();
          return scale;
        };
        return scale;
      }
    };

    const renderer = createD3DendrogramRenderer(fakeD3, { nodeRadius: 2 });
    const svg = spec.show(renderer);

    expect(svg.tagName).toBe('SVG');
    expect(svg.getAttribute('width')).toBe(String(spec.config.width));

    const circles = svg.querySelectorAll('circle');
    expect(circles.length).toBe(model.n + model.dendrogram.length);

    const lines = svg.querySelectorAll('line');
    expect(lines.length).toBe(model.dendrogram.length * 3);

    const labels = svg.querySelectorAll('text');
    expect(labels.length).toBe(model.n);
  });

  it('createD3DendrogramRenderer respects orientation overrides', () => {
    const model = fit(sampleData);
    const spec = plotHCA(model);
    const fakeD3 = {
      scaleLinear() {
        let domain = [0, 1];
        let range = [0, 1];
        const scale = (value) => {
          const span = domain[1] - domain[0];
          if (span === 0) return range[0];
          const t = (value - domain[0]) / span;
          return range[0] + t * (range[1] - range[0]);
        };
        scale.domain = (values) => {
          domain = values.slice();
          return scale;
        };
        scale.range = (values) => {
          range = values.slice();
          return scale;
        };
        return scale;
      }
    };

    const renderer = createD3DendrogramRenderer(fakeD3);
    const horizontalSvg = spec.show(renderer, {
      config: { orientation: 'horizontal', width: 500, height: 300 }
    });

    expect(horizontalSvg.tagName).toBe('SVG');
    expect(horizontalSvg.getAttribute('width')).toBe('500');
    expect(horizontalSvg.getAttribute('height')).toBe('300');

    const leafCircles = Array.from(horizontalSvg.querySelectorAll('circle'))
      .filter(circle => {
        const id = Number(circle.dataset.nodeId);
        return Number.isFinite(id) && id < model.n;
      });

    const uniqueXPositions = new Set(leafCircles.map(circle => circle.getAttribute('cx')));
    expect(uniqueXPositions.size).toBe(1);
  });
});
