/**
 * HCA visualization helpers
 * Dendrogram configuration
 */

import { attachTreeShow } from './show.js';

/**
 * Generate dendrogram data structure
 * @param {Object} result - HCA result from ml.hca.fit()
 * @returns {Object} Dendrogram tree structure
 */
export function plotHCA(result) {
  const { dendrogram, n } = result;
  
  // Create initial leaf nodes
  const nodes = Array.from({ length: n }, (_, i) => ({
    id: i,
    height: 0,
    isLeaf: true,
    children: null,
    size: 1
  }));
  
  // Build tree from merges
  const tree = { nodes: [...nodes], merges: [] };
  
  dendrogram.forEach((merge, idx) => {
    const { cluster1, cluster2, distance, size } = merge;
    
    tree.merges.push({
      id: n + idx,
      cluster1: cluster1.slice(),
      cluster2: cluster2.slice(),
      height: distance,
      size,
      children: [cluster1[0], cluster2[0]]
    });
  });
  
  return attachTreeShow({
    type: 'dendrogram',
    data: tree,
    config: {
      width: 640,
      height: 400,
      linkage: result.linkage,
      orientation: 'vertical'
    }
  });
}

/**
 * Convert dendrogram to layout coordinates
 * @param {Object} dendrogramData - Result from plotHCA
 * @param {Object} options - {width, height, orientation}
 * @returns {Object} Layout with coordinates
 */
export function dendrogramLayout(dendrogramData, { 
  width = 640, 
  height = 400,
  orientation = 'vertical'
} = {}) {
  const { data } = dendrogramData;
  const { nodes, merges } = data;
  
  const layout = {
    nodes: [],
    links: []
  };
  
  // Assign x positions to leaves
  const spacing = width / (nodes.length + 1);
  nodes.forEach((node, i) => {
    layout.nodes.push({
      id: node.id,
      x: (i + 1) * spacing,
      y: 0,
      isLeaf: true
    });
  });
  
  // Position internal nodes
  merges.forEach((merge, idx) => {
    const { cluster1, cluster2, height } = merge;
    
    // Find positions of children
    const child1Pos = layout.nodes.find(n => cluster1.includes(n.id));
    const child2Pos = layout.nodes.find(n => cluster2.includes(n.id));
    
    if (child1Pos && child2Pos) {
      const x = (child1Pos.x + child2Pos.x) / 2;
      const y = height;
      
      layout.nodes.push({
        id: nodes.length + idx,
        x,
        y,
        isLeaf: false
      });
      
      // Add links
      layout.links.push(
        { source: { x: child1Pos.x, y: child1Pos.y }, target: { x: child1Pos.x, y: height } },
        { source: { x: child1Pos.x, y: height }, target: { x: child2Pos.x, y: height } },
        { source: { x: child2Pos.x, y: height }, target: { x: child2Pos.x, y: child2Pos.y } }
      );
    }
  });
  
  return {
    type: 'dendrogramLayout',
    width,
    height,
    data: layout
  };
}
