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
    leafLabel = (node) => `${node.id}`,
    distanceScale = 'linear',
    distanceEpsilon = 1e-6,
    labelOrientation = 'auto',
    labelPadding = 12
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

    const distances = data.merges.map((merge) => merge.height ?? merge.distance ?? 0);
    const { transformDistance, maxTransformedDistance } = buildDistanceTransform(
      distances,
      distanceScale,
      distanceEpsilon
    );

    let distanceToOffset;
    if (modules.scaleLinear) {
      distanceToOffset = modules
        .scaleLinear()
        .domain([0, Math.max(maxTransformedDistance, 1)])
        .range([0, baseDistanceRange]);
    } else {
      distanceToOffset = (value) => {
        if (!maxTransformedDistance) return 0;
        return (value / maxTransformedDistance) * baseDistanceRange;
      };
    }

    const leafNodes = [];
    const internalNodes = [];
    const segments = [];

    const clusterMap = new Map();
    const leaves = data.nodes;
    const leafCount = leaves.length;

    leaves.forEach((leaf, index) => {
      const node = {
        id: leaf.id,
        x: 0,
        y: 0,
        distance: 0,
        isLeaf: true,
        children: []
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

      const parent = {
        id: merge.id ?? leafCount + idx,
        distance: clusterDistance,
        isLeaf: false,
        size,
        children: [node1, node2]
      };

      node1.parent = parent;
      node2.parent = parent;
      internalNodes.push(parent);
      const combinedKey = getKey([...cluster1, ...cluster2].sort((a, b) => a - b));
      clusterMap.set(combinedKey, parent);
    });

    const root = internalNodes.length > 0 ? internalNodes[internalNodes.length - 1] : leafNodes[0];

    let leafCursor = 0;
    const totalLeaves = Math.max(leafNodes.length, 1);
    const leafStepX = totalLeaves > 1 ? availableWidth / (totalLeaves - 1) : 0;
    const leafStepY = totalLeaves > 1 ? availableHeight / (totalLeaves - 1) : 0;

    function assignPositions(node) {
      if (node.isLeaf || !node.children || node.children.length === 0) {
        if (orientation === 'vertical') {
          node.x = totalLeaves > 1
            ? margins.left + leafStepX * leafCursor
            : margins.left + availableWidth / 2;
          node.y = baseCoordinate;
        } else {
          node.x = baseCoordinate;
          node.y = totalLeaves > 1
            ? margins.top + leafStepY * leafCursor
            : margins.top + availableHeight / 2;
        }
        leafCursor += 1;
        return;
      }

      node.children.forEach(assignPositions);

      if (orientation === 'vertical') {
        node.x = node.children.reduce((sum, child) => sum + child.x, 0) / node.children.length;
        node.y = baseCoordinate - distanceToOffset(transformDistance(node.distance));
      } else {
        node.x = baseCoordinate - distanceToOffset(transformDistance(node.distance));
        node.y = node.children.reduce((sum, child) => sum + child.y, 0) / node.children.length;
      }
    }

    if (root) {
      assignPositions(root);
    }

    function collectSegments(node) {
      if (!node || !node.children || node.children.length === 0) {
        return;
      }

      if (orientation === 'vertical') {
        const childXs = node.children.map((child) => child.x);
        const minX = Math.min(...childXs);
        const maxX = Math.max(...childXs);

        node.children.forEach((child) => {
          segments.push({ x1: child.x, y1: child.y, x2: child.x, y2: node.y });
        });
        segments.push({ x1: minX, y1: node.y, x2: maxX, y2: node.y });
      } else {
        const childYs = node.children.map((child) => child.y);
        const minY = Math.min(...childYs);
        const maxY = Math.max(...childYs);

        node.children.forEach((child) => {
          segments.push({ x1: child.x, y1: child.y, x2: node.x, y2: child.y });
        });
        segments.push({ x1: node.x, y1: minY, x2: node.x, y2: maxY });
      }

      node.children.forEach(collectSegments);
    }

    if (root) {
      collectSegments(root);
    }

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

        applyLabelPositioning({
          text,
          node,
          orientation,
          labelPadding,
          resolvedOrientation: resolveLabelOrientation(labelOrientation, orientation)
        });
        labelGroup.appendChild(text);
      });
    }

    return svg;
  };
}

export default createD3DendrogramRenderer;

function buildDistanceTransform(distances, scaleOption, epsilon) {
  const cleanDistances = distances.filter((value) => Number.isFinite(value) && value >= 0);
  const hasDistances = cleanDistances.length > 0;

  let transform;
  if (typeof scaleOption === 'function') {
    transform = (distance) => {
      const result = scaleOption(distance);
      return Number.isFinite(result) ? Math.max(0, result) : 0;
    };
  } else {
    switch (scaleOption) {
      case 'log10':
      case 'log':
        transform = (distance) => Math.log10(Math.max(0, distance) + 1);
        break;
      case 'log1p':
        transform = (distance) => Math.log1p(Math.max(0, distance));
        break;
      case 'sqrt':
        transform = (distance) => Math.sqrt(Math.max(0, distance));
        break;
      default:
        transform = (distance) => Math.max(0, distance);
    }
  }

  if (!hasDistances) {
    return {
      transformDistance: (distance) => Math.max(0, transform(distance)),
      maxTransformedDistance: 0
    };
  }

  const transformed = cleanDistances.map((distance) => transform(distance));
  const maxValue = Math.max(...transformed);
  const adjustedMax = Math.max(maxValue, epsilon, 0);

  return {
    transformDistance: (distance) => Math.max(0, transform(distance)),
    maxTransformedDistance: adjustedMax
  };
}

function resolveLabelOrientation(option, orientation) {
  if (!option || option === 'auto') {
    return orientation === 'horizontal' ? 'horizontal' : 'vertical';
  }
  return option;
}

function applyLabelPositioning({
  text,
  node,
  orientation,
  labelPadding,
  resolvedOrientation
}) {
  if (orientation === 'vertical') {
    const x = Number(node.x.toFixed(2));
    const y = Number((node.y + labelPadding).toFixed(2));

    text.setAttribute('x', x.toString());
    text.setAttribute('y', y.toString());

    if (resolvedOrientation === 'vertical') {
      text.setAttribute('transform', `rotate(-90 ${x} ${y})`);
      text.setAttribute('text-anchor', 'end');
      text.setAttribute('dominant-baseline', 'middle');
    } else {
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'hanging');
    }
  } else {
    const x = Number((node.x + labelPadding).toFixed(2));
    const y = Number(node.y.toFixed(2));

    text.setAttribute('x', x.toString());
    text.setAttribute('y', y.toString());
    text.setAttribute('dominant-baseline', 'middle');

    if (resolvedOrientation === 'vertical') {
      text.setAttribute('transform', `rotate(-90 ${x} ${y})`);
      text.setAttribute('text-anchor', 'middle');
    } else {
      text.setAttribute('text-anchor', 'start');
    }
  }
}
