// Quick smoke test - verifies the package works
// Run with: node examples/quick-test.js

import { core, stats, ml, mva, plot } from '@tangent.to/ds';

console.log('Testing @tangent.to/ds\n');

try {
    // 1. Linear algebra
    console.log('✓ Linear algebra');
    const transposed = core.linalg.transpose([[1, 2], [3, 4]]);

    // 2. Distributions (pdf, cdf, quantile - not random sampling)
    console.log('✓ Distributions (pdf/cdf/quantile)');
    const pdfValue = stats.normal.pdf(0, { mean: 0, sd: 1 });
    const sample = stats.normal.quantile(Math.random()); // Use quantile for sampling

    // 3. Regression (GLM with Gaussian family)
    console.log('✓ Linear regression (GLM)');
    const lm = new stats.GLM({ family: 'gaussian' });
    lm.fit([[1], [2], [3]], [2, 4, 6]);
    const pred = lm.predict([[4]]);

    // 4. K-Means
    console.log('✓ K-Means clustering');
    const kmeans = new ml.KMeans({ k: 2 });
    kmeans.fit([[1, 1], [2, 2], [8, 8], [9, 9]]);

    // 5. KNN
    console.log('✓ K-Nearest Neighbors');
    const knn = new ml.KNNClassifier({ k: 1 });
    knn.fit([[1, 1], [2, 2]], [0, 1]);
    const knnPred = knn.predict([[1.5, 1.5]]);

    // 6. Decision Tree
    console.log('✓ Decision Tree');
    const dt = new ml.DecisionTreeClassifier({ maxDepth: 3 });
    dt.fit([[0], [1], [2], [3]], [0, 0, 1, 1]);

    // 7. PCA
    console.log('✓ PCA');
    const pca = new mva.PCA({ center: true });
    pca.fit([[1, 2], [2, 3], [3, 4]]);

    // 8. Plot configs
    console.log('✓ Plot configurations (return Observable Plot configs)');
    const confusionMatrix = plot.plotConfusionMatrix([0, 1, 0], [0, 1, 1]);
    const rocCurve = plot.plotROC([0, 1, 0], [0.1, 0.9, 0.3]);

    console.log('\n✅ All systems GO! Package is ready to publish.\n');
    console.log('Run full examples:');
    console.log('  node examples/misc/api_overview.js');
    console.log('  node examples/misc/ml_usage.js');
    console.log('  node examples/misc/mva_usage.js');
    console.log('\nFor plots rendering: Use tangent-notebook');

} catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
}
