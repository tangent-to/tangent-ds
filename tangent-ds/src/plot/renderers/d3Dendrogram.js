const SVG_NS = 'http://www.w3.org/2000/svg';

function normalizeD3(d3) {
  if (!d3) {
    return {};
  }
  const scaleLinear = typeof d3.scaleLinear === 'function' ? d3.scaleLinear : null;
  return { scaleLinear };
}

function resolveMargins(config = {}, orientation) {
  const fallback = config.margin ?? 32;
  return {
    top: config.marginTop ?? fallback,
    right: config.marginRight ?? fallback,
    bottom: config.marginBottom ?? (orientation === 'vertical' ? Math.max(48, fallback) : fallback),
    left: config.marginLeft ?? (orientation === 'horizontal' ? Math.max(48, fallback) : fallback)
  };
}

function makeClusterKey(indices = []) {
  return indices
    .slice()
    .sort((a, b) => a - b)
    .join('|');
}

function createSvgElement(tag) {
  return document.createElementNS(SVG_NS, tag);
}

/**
 * Build a dendrogram renderer that relies on user-supplied D3 modules for scaling.
 * The returned function is compatible with the `.show(renderer)` helper emitted by plotHCA.
 *
 * Usage:
 *   import { plotHCA } from '@tangent.to/ds/plot';
 *   import { createD3DendrogramRenderer } from '@tangent.to/ds/plot/renderers/d3Dendrogram.js';
 *   const spec = plotHCA(model);
 *   const svg = spec.show(createD3DendrogramRenderer(d3));
 *
 * @param {Object} d3 - D3 namespace (only `scaleLinear` is used if available)
 * @param {Object} options - Renderer options
 * @returns {Function} Renderer function accepted by config.show()
 */
export function createD3DendrogramRenderer(
  d3,
  {
    nodeRadius = 4,
    leafFill = '#ff6b6b',
    internalFill = '#555',
    linkStroke = '#999',
    fontFamily = 'system-ui, sans-serif',
    fontSize = 11,
    leafLabel = (node) => `${node.id}`
  } = {}
) {
  const modules = normalizeD3(d3);

  return function renderDendrogram({ data, config = {} }) {
    if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.merges)) {
      throw new Error('createD3DendrogramRenderer: expected dendrogram data with nodes and merges.');
    }

    const orientation = config.orientation === 'horizontal' ? 'horizontal' : 'vertical';
    const width = typeof config.width === 'number' ? config.width : 640;
    const height = typeof config.height === 'number' ? config.height : 400;
    const margins = resolveMargins(config, orientation);

    const availableWidth = Math.max(0, width - margins.left - margins.right);
    const availableHeight = Math.max(0, height - margins.top - margins.bottom);
    const baseDistanceRange = orientation === 'vertical' ? availableHeight : availableWidth;

    const maxDistance = data.merges.reduce(
      (max, merge) => Math.max(max, merge.height ?? merge.distance ?? 0),
      0
    );

    let distanceToOffset;
    if (modules.scaleLinear) {
      distanceToOffset = modules
        .scaleLinear()
        .domain([0, Math.max(maxDistance, 1)])
        .range([0, baseDistanceRange]);
    } else {
      distanceToOffset = (distance) => {
        if (!maxDistance) return 0;
        return (distance / maxDistance) * baseDistanceRange;
      };
    }

    const leafNodes = [];
    const internalNodes = [];
    const segments = [];

    const clusterMap = new Map();
    const leaves = data.nodes;
    const leafCount = leaves.length;
    const spacingDenominator = Math.max(leafCount - 1, 1);

    const leafPositioner = orientation === 'vertical'
      ? (index) => ({
        x: margins.left + (availableWidth * index) / spacingDenominator,
        y: margins.top + availableHeight
      })
      : (index) => ({
        x: margins.left + availableWidth,
        y: margins.top + (availableHeight * index) / spacingDenominator
      });

    leaves.forEach((leaf, index) => {
      const position = leafPositioner(index);
      const node = {
        id: leaf.id,
        x: position.x,
        y: position.y,
        distance: 0,
        isLeaf: true
      };
      leafNodes.push(node);
      clusterMap.set(makeClusterKey([leaf.id]), node);
    });

    const mergeKeyCache = new Map();
    const getKey = (indices) => {
      const cacheKey = indices.join(',');
      if (!mergeKeyCache.has(cacheKey)) {
        mergeKeyCache.set(cacheKey, makeClusterKey(indices));
      }
      return mergeKeyCache.get(cacheKey);
    };

    const baseCoordinate = orientation === 'vertical'
      ? margins.top + availableHeight
      : margins.left + availableWidth;

    data.merges.forEach((merge, idx) => {
      const { cluster1 = [], cluster2 = [], height, distance, size } = merge;
      const clusterDistance = height ?? distance ?? 0;
      const key1 = getKey(cluster1);
      const key2 = getKey(cluster2);
      const node1 = clusterMap.get(key1);
      const node2 = clusterMap.get(key2);

      if (!node1 || !node2) {
        throw new Error('createD3DendrogramRenderer: malformed dendrogram merge structure.');
      }

      const offset = distanceToOffset(clusterDistance);
      const parent = {
        id: merge.id ?? leafCount + idx,
        distance: clusterDistance,
        isLeaf: false,
        size
      };

      if (orientation === 'vertical') {
        parent.x = (node1.x + node2.x) / 2;
        parent.y = baseCoordinate - offset;

        segments.push(
          { x1: node1.x, y1: node1.y, x2: node1.x, y2: parent.y },
          { x1: node2.x, y1: node2.y, x2: node2.x, y2: parent.y },
          { x1: node1.x, y1: parent.y, x2: node2.x, y2: parent.y }
        );
      } else {
        parent.x = baseCoordinate - offset;
        parent.y = (node1.y + node2.y) / 2;

        segments.push(
          { x1: node1.x, y1: node1.y, x2: parent.x, y2: node1.y },
          { x1: node2.x, y1: node2.y, x2: parent.x, y2: node2.y },
          { x1: parent.x, y1: node1.y, x2: parent.x, y2: node2.y }
        );
      }

      internalNodes.push(parent);
      const combinedKey = getKey([...cluster1, ...cluster2].sort((a, b) => a - b));
      clusterMap.set(combinedKey, parent);
    });

    const svg = createSvgElement('svg');
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('role', 'img');
    svg.style.fontFamily = fontFamily;
    svg.style.fontSize = `${fontSize}px`;

    const group = createSvgElement('g');
    svg.appendChild(group);

    const linkGroup = createSvgElement('g');
    linkGroup.setAttribute('fill', 'none');
    linkGroup.setAttribute('stroke', linkStroke);
    linkGroup.setAttribute('stroke-width', '1.25');
    group.appendChild(linkGroup);

    segments.forEach((segment) => {
      const line = createSvgElement('line');
      line.setAttribute('x1', segment.x1.toFixed(2));
      line.setAttribute('y1', segment.y1.toFixed(2));
      line.setAttribute('x2', segment.x2.toFixed(2));
      line.setAttribute('y2', segment.y2.toFixed(2));
      linkGroup.appendChild(line);
    });

    const nodeGroup = createSvgElement('g');
    nodeGroup.setAttribute('stroke', '#fff');
    nodeGroup.setAttribute('stroke-width', '0.75');
    group.appendChild(nodeGroup);

    const allNodes = [...leafNodes, ...internalNodes];
    allNodes.forEach((node) => {
      const circle = createSvgElement('circle');
      circle.setAttribute('cx', node.x.toFixed(2));
      circle.setAttribute('cy', node.y.toFixed(2));
      circle.setAttribute('r', node.isLeaf ? String(nodeRadius) : String(nodeRadius * 0.85));
      circle.setAttribute('fill', node.isLeaf ? leafFill : internalFill);
      circle.dataset.nodeId = String(node.id);
      nodeGroup.appendChild(circle);
    });

    if (typeof leafLabel === 'function') {
      const labelGroup = createSvgElement('g');
      labelGroup.setAttribute('fill', '#333');
      group.appendChild(labelGroup);

      leafNodes.forEach((node) => {
        const label = leafLabel(node);
        if (label === null || label === undefined || label === '') {
          return;
        }
        const text = createSvgElement('text');
        text.textContent = String(label);

        if (orientation === 'vertical') {
          text.setAttribute('x', node.x.toFixed(2));
          text.setAttribute('y', (node.y + nodeRadius * 3).toFixed(2));
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('dominant-baseline', 'hanging');
        } else {
          text.setAttribute('x', (node.x + nodeRadius * 3).toFixed(2));
          text.setAttribute('y', node.y.toFixed(2));
          text.setAttribute('dominant-baseline', 'middle');
          text.setAttribute('text-anchor', 'start');
        }
        labelGroup.appendChild(text);
      });
    }

    return svg;
  };
}

export default createD3DendrogramRenderer;
