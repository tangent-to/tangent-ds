TODO
- planned features
    - ml: y can take multiple targets (one model per target, for now)
    - implement tensorflowjs
    - implement xgboost
    - implement Gaussian processes inspired by https://github.com/jmonlabs/jmon-algo/tree/main/src/algorithms/generative/gaussian-processes, then generate python tests and examples
    - implement CCA

DOING


DONE
- implement Tukey HSD test
- scikit-learn's-like API across the whole pakcage, ml, mvs and plot
- fix logistic regression
- fix kmeans
- test Python similarity
- implement KNN, decision trees, random rest and generalized additive models (both in classification and regression)
- implement python tests and examples for KNN, decision trees, random forest and generalized additive models
- fix PCA, implement SVD
- review train, validation, preprocessing, tunig and interpret
- review plot, one ordiplot module for ordinations (PCA, LDA, RDA), add plot for feature importance, add ROC curves, add other ploting facilitators you might think useful...
- model persistance in json format
- the workflow publishes to npm and pushes the site on tag
Before the workflow can run successfully, you'll need to:
- Add NPM_TOKEN secret to GitHub repository settings (see RELEASE.md for details)
- Ensure GitHub Pages is enabled in repository settings
- Test by pushing a tag: git tag v0.7.1 && git push origin
- review arquero dependency
- implement CCA
- implement scaling
- check if PCA, LDA, CCA and RDA can predict the scores
- fix loading labels
- fix scaling
- fix on-hot encoding
- auto triplot
- scale a center to true by default
- Set up TypeDoc extraction and inject it into Docusaurus properly - to be tested
- implement Ward hierarchical clustering
- In ordination classes, the .show() method indicates the pltting library as .show(Plot) to avoid including the plotting library in the package. Could you do it similarly throught mva.js and ther modules? For instance, add the plot library object as argument to hierarchical clustering like .show(d3-dendrogram)
- rewrite tangent-notebook and Obsrevable tutorials
- reboot scaling
- fix LDA
- update docu
- recipe interface
- check the noteboks
