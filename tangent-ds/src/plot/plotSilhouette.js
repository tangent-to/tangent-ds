import { attachShow } from './show.js';
import { silhouetteSamples, silhouetteByCluster } from '../ml/silhouette.js';

function ensureSamples(options) {
  if (options && Array.isArray(options.samples)) {
    return options.samples;
  }
  if (!options || !options.data || !options.labels) {
    throw new Error(
      'plotSilhouette requires either { samples } or { data, labels } options.'
    );
  }
  const samples = silhouetteSamples(options.data, options.labels);
  if (options.sorted !== false) {
    return samples.slice().sort((a, b) => b.silhouette - a.silhouette);
  }
  return samples;
}

function ensureClusters(options, samples) {
  if (options && Array.isArray(options.clusters)) {
    return options.clusters;
  }
  if (options && options.data && options.labels) {
    return silhouetteByCluster(options.data, options.labels);
  }
  if (samples && samples.length) {
    const groups = new Map();
    samples.forEach((sample) => {
      const bucket = groups.get(sample.cluster) || [];
      bucket.push(sample);
      groups.set(sample.cluster, bucket);
    });
    return [...groups.entries()].map(([cluster, entries]) => ({
      cluster,
      samples: entries
    }));
  }
  return [];
}

function buildPerSampleDataset(samples) {
  return samples.map((entry, index) => ({
    index,
    cluster: String(entry.cluster),
    silhouette: entry.silhouette,
    a: entry.a,
    b: entry.b
  }));
}

function buildClusterSummaryDataset(clusters) {
  return clusters.map((clusterInfo) => ({
    cluster: String(clusterInfo.cluster),
    average: clusterInfo.average
  }));
}

/**
 * Generate silhouette plot configuration displaying per-sample scores.
 * Accepts either precomputed samples or raw data/labels for convenience.
 *
 * @param {Object} options
 * @param {Array} [options.samples] - Output from ml.silhouette.silhouetteSamples()
 * @param {Array} [options.data] - Data matrix used to compute silhouette scores
 * @param {Array} [options.labels] - Cluster labels for each observation
 * @param {boolean} [options.sorted=true] - Whether to sort samples by silhouette desc
 * @param {number} [options.minSilhouette=-1] - Minimum silhouette value displayed
 * @param {number} [options.maxSilhouette=1] - Maximum silhouette value displayed
 * @param {Object} [options.clusterOptions] - Options for cluster summary inset
 * @returns {Object} Observable Plot-compatible configuration with `.show()`
 */
export function plotSilhouette(
  options = {},
  {
    width = 720,
    height = 420,
    minSilhouette = -1,
    maxSilhouette = 1,
    clusterInsetWidth = 160,
    clusterInsetHeight = 160,
    showAverageLines = true
  } = {}
) {
  const samples = ensureSamples(options);
  const clusters = ensureClusters(options, samples);

  const values = buildPerSampleDataset(samples);
  const clusterSummary = clusters && clusters.length
    ? buildClusterSummaryDataset(clusters)
    : null;

  const config = {
    type: 'silhouette',
    width,
    height,
    data: {
      values,
      ...(clusterSummary ? { clusterSummary } : {})
    },
    axes: {
      x: {
        label: 'Silhouette score',
        domain: [minSilhouette, maxSilhouette],
        grid: true
      },
      y: { label: 'Sample index', tickFormat: '' }
    },
    marks: [
      {
        type: 'barX',
        data: 'values',
        x: 'silhouette',
        y: 'index',
        fill: 'cluster'
      },
      {
        type: 'ruleX',
        x: 0,
        stroke: '#666',
        strokeDasharray: '4,4'
      }
    ],
    annotations: clusterSummary
      ? [
          {
            type: 'inset',
            width: clusterInsetWidth,
            height: clusterInsetHeight,
            align: 'top-right',
            padding: 8,
            config: {
              type: 'silhouetteClusterSummary',
              width: clusterInsetWidth,
              height: clusterInsetHeight,
              data: { clusterSummary },
              axes: {
                x: { label: 'Cluster', tickRotate: -45 },
                y: { label: 'Average silhouette', domain: [minSilhouette, maxSilhouette] }
              },
              marks: [
                {
                  type: 'barY',
                  data: 'clusterSummary',
                  x: 'cluster',
                  y: 'average',
                  fill: 'steelblue'
                },
                ...(showAverageLines
                  ? [
                      {
                        type: 'ruleY',
                        y: 'average',
                        stroke: '#333',
                        strokeWidth: 1
                      }
                    ]
                  : [])
              ]
            }
          }
        ]
      : []
  };

  return attachShow(config);
}

export default plotSilhouette;
