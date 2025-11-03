import { prepareX } from "../core/table.js";

function pairwiseDistances(data) {
  const n = data.length;
  const distances = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dist = euclideanDistance(data[i], data[j]);
      distances[i][j] = dist;
      distances[j][i] = dist;
    }
  }

  return distances;
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function ensureDataMatrix(X) {
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    (X.data || X.columns)
  ) {
    const prepared = prepareX({
      data: X.data,
      columns: X.columns || X.X,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return prepared.X;
  }

  if (Array.isArray(X)) {
    if (!Array.isArray(X[0])) {
      return X.map((val) => [val]);
    }
    return X;
  }

  throw new Error(
    "silhouetteSamples expects an array of rows or an options object { data, columns }."
  );
}

export function silhouetteSamples(X, labels) {
  const data = ensureDataMatrix(X);
  const n = data.length;

  if (!Array.isArray(labels) || labels.length !== n) {
    throw new Error("silhouetteSamples: labels must be an array with length matching data.");
  }

  const clusters = new Map();
  labels.forEach((label, index) => {
    if (!clusters.has(label)) {
      clusters.set(label, []);
    }
    clusters.get(label).push(index);
  });

  const uniqueClusters = [...clusters.keys()];
  if (uniqueClusters.length < 2) {
    throw new Error("silhouetteSamples requires at least two clusters.");
  }

  const distances = pairwiseDistances(data);

  return data.map((_, index) => {
    const label = labels[index];
    const ownCluster = clusters.get(label);

    let a = 0;
    if (ownCluster.length > 1) {
      let sum = 0;
      for (const otherIndex of ownCluster) {
        if (otherIndex === index) continue;
        sum += distances[index][otherIndex];
      }
      a = sum / (ownCluster.length - 1);
    }

    let b = Infinity;
    for (const clusterLabel of uniqueClusters) {
      if (clusterLabel === label) continue;
      const clusterMembers = clusters.get(clusterLabel);
      let sum = 0;
      for (const otherIndex of clusterMembers) {
        sum += distances[index][otherIndex];
      }
      const avg = sum / clusterMembers.length;
      if (avg < b) {
        b = avg;
      }
    }

    const denom = Math.max(a, b);
    const silhouette = denom === 0 ? 0 : (b - a) / denom;

    return {
      index,
      cluster: label,
      a,
      b,
      silhouette
    };
  });
}

export function silhouetteByCluster(X, labels) {
  const samples = silhouetteSamples(X, labels);
  const clusters = new Map();

  samples.forEach((entry) => {
    const arr = clusters.get(entry.cluster) || [];
    arr.push(entry);
    clusters.set(entry.cluster, arr);
  });

  const sorted = [...clusters.entries()].map(([cluster, entries]) => {
    const ordered = entries.slice().sort((a, b) => b.silhouette - a.silhouette);
    return {
      cluster,
      samples: ordered,
      average: ordered.reduce((sum, entry) => sum + entry.silhouette, 0) / ordered.length
    };
  });

  sorted.sort((a, b) => b.average - a.average);

  return sorted;
}

export default {
  silhouetteSamples,
  silhouetteByCluster
};
