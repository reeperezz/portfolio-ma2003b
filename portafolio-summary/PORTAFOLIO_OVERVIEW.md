# Portfolio Overview — Multivariate Methods Integrative Document

This document provides an integrative view of the three multivariate techniques used across the portfolio cases: Exploratory Factor Analysis (EFA), Discriminant Analysis (LDA / QDA) and Clustering (Hierarchical + K-Means; PCA used for projection/visualization). It explains how the methods relate, the business questions they answer, when to use each technique, and a conceptual map describing their relationships and typical pipelines.

---

## Executive Summary 

- The portfolio contains three case studies that illustrate complementary multivariate approaches:
	- Case 1 — Factor Analysis (customer satisfaction): discover latent drivers from many correlated survey items.
	- Case 2 — Discriminant Analysis (credit risk): supervised classification that separates labeled groups like defaulters vs reliable borrowers.
	- Case 3 — Clustering + PCA (retail segmentation): unsupervised customer segmentation for targeted marketing.
- These methods work best as part of a pipeline: dimension reduction (EFA / PCA) helps interpretability and avoids multicollinearity; clustering and discriminant methods answer segmentation and classification needs; methods often feed into each other (e.g., factors used as inputs for discriminant models or clusters used for supervised learning).

---

## 1) How the three methods relate to each other 

All three methods are multivariate techniques used to summarize and extract structure from datasets with many variables, yet they serve different analytical purposes and can be combined as part of a modeling pipeline:

- **Dimension reduction methods (EFA & PCA)**
	- EFA (Exploratory Factor Analysis): seeks latent constructs that explain correlated variables (useful especially with survey or psychometric data). EFA is explicitly aimed at interpreting the latent meaning of variable groups.
	- PCA (Principal Component Analysis): reduces dimensionality by finding orthogonal directions of maximal variance; used for visualization and sometimes pre-processing (not aimed at extracting meaningful latent constructs).

- **Clustering (hierarchical, k-means)**
	- Unsupervised grouping to identify natural segments when no labels are available. Clustering benefits from dimension reduction (PCA/EFA) to remove noise and keep the most informative features.

- **Discriminant Analysis (LDA/QDA)**
	- Supervised classification to separate known groups (labels). Discriminant analysis can be used to validate whether groups (for example discovered via clustering) are separable using available features.

Inter-relationships:
- Use PCA/EFA before clustering or discriminant models when you have many correlated features or want more interpretable, lower-dimensional inputs.
- Convert cluster assignments into labels to train supervised discriminant models for operational classification and automation.
- Use discriminant analysis to test the separability of clusters and to produce linear/quadratic boundaries for operational classification.

---

## 2) What types of business questions each method answers 

- **Factor Analysis (Case 1 — Customer Satisfaction)**
	- What underlying dimensions explain many correlated survey items?
	- Which latent drivers (factors) are the strongest predictors of commercial outcomes (NPS, renewals, revenue)?
	- How can we reduce many questionnaire items into a few actionable constructs for reporting and interventions?

- **Discriminant Analysis (Case 2 — Credit Risk)**
	- Can we classify a loan applicant into 'default' or 'non-default' classes with available features?
	- Which combination of indicators best separate the risk groups? (Interpretability for regulatory and business use.)
	- How to operationalize a decision boundary for automated approvals (risk screening)?

- **Clustering + PCA (Case 3 — Customer Segmentation)**
	- What natural segments exist in the customer base based on behavior, recency, and spending?
	- Which segments should we target with specific marketing strategies (re-activation, loyalty, conversion)?
	- How to identify segment-specific KPIs and expected revenue impact from targeted campaigns?

---

## 3) When to use each multivariate technique — Practical guidance 

Use the technique that matches the project goal and data structure. Here are practical rules of thumb:

- **Exploratory Factor Analysis — Use when:**
	- Data consist of many correlated items measuring the same construct (e.g., survey items).
	- You want interpretable latent constructs (factors) and to reduce multicollinearity in downstream models.
	- Pre-conditions: adequate sample size, KMO measure ≥ 0.6, significant Bartlett’s test of sphericity.
	- Result: factor scores that can be used as predictors or as concise reporting metrics.

- **Principal Component Analysis — Use when:**
	- Your aim is to compress the data for visualization or to reduce dimensionality for clustering/classification without needing interpretable latent constructs.
	- PCA commonly supports cluster visualizations (e.g., 2D scatterplots) and helps detect structure.

- **Clustering (Hierarchical / K-Means) — Use when:**
	- You do not have labels but need to discover groups (segments) with similar behavior or characteristics.
	- Useful for marketing segmentation, customer profiling, and targeted action design.
	- Pre-steps: scale and normalize features; remove collinear/noisy variables, and consider PCA/EFA first.
	- Validate: silhouette score, elbow method, stability across seeds, and silhouette or silhouette + elbow for robust k.

- **Discriminant Analysis (LDA / QDA) — Use when:**
	- You have labeled groups and your objective is classification with interpretable decision boundaries.
	- LDA is preferable when class covariance matrices are similar (homoscedasticity).
	- QDA is appropriate when class covariances differ and you want nonlinear boundaries.
	- Discriminant analysis also helps interpret which features move groups apart.

---

## 4) Conceptual Map — Relationships between methods (English) 

Below is a compact visual map showing how these methods usually connect in a data analysis pipeline. Replace 'X' with EFA/PCA as needed depending on the data and the project goal.

1. Start: Raw features (many variables, e.g. survey items or behavioral metrics)
	 - If the dataset has many correlated items OR you need interpretable latent constructs: apply EFA -> factor scores
	 - If the dataset has high dimensionality but you only want to visualize or compress variance: apply PCA -> principal components
2. Choose analysis depending on question:
	 - Goal: Understand latent drivers / create concise reporting metrics -> Use factor scores (EFA)
	 - Goal: Discover segments without labels -> Use clustering (K-Means, Hierarchical) on original features or PCs/factors
	 - Goal: Build a classifier for existing labels -> Use discriminant analysis (LDA / QDA) using original features, PCs, or factor scores
3. Optional combinational flows:
	 - PCA/EFA -> Clustering -> Clusters (no labels) -> Use cluster assignments as labels -> Build discriminant model to classify new observations quickly
	 - EFA -> Regression / Predictive models -> Output business KPIs (e.g., overall satisfaction / revenue)
	 - Clustering -> Segment-based experiments -> A/B test targeted campaigns

ASCII conceptual map (simple):

Raw Data (many features)
 ├─> EFA (factors) ──> Factor Scores ──> Predictive Models / Reports
 ├─> PCA (principal components) ──> Visualization & Dim. Reduction ──> Clustering/Models
 └─> Clustering (hierarchical/k-means) ──> Customer Segments
		 └─> Cluster labels ──> Discriminant Analysis (classifier) or Actionable Campaigns

---

## 5) Quick Decision Tree — Which method should I use? 

- If your objective is to reduce many survey items into interpretable factors: use **EFA**.
- If you need to discover market/customer segments and there are no labels: use **clustering** (and PCA for visualization).
- If you want a classifier for labeled outcomes (like default vs non-default): use **discriminant analysis** (LDA/QDA). Use LDA for similar-group covariances, QDA when variance differs.
- Combine methods when appropriate: use EFA/PCA to reduce noise and improve performance/interpretablity of clustering or discriminant models.

---

## 6) Case study references in this repo 

- `case-01-factor-analysis` (Customer Satisfaction) — EFA used for latent drivers and as predictors for regressions. See `costomer_satisfaction.ipynb` and `visualizations`.
- `case-02-discriminant-analysis` (Credit Risk) — LDA & QDA applied to binary default classification. See `credit_risk_analysis.ipynb`.
- `caso-03-megamart-customer` (Retail Segmentation) — Hierarchical clustering and K-Means with PCA for cluster visualization. See `retail.ipynb`.

---

## 7) Short recommendations for practitioners (Actionable checklist) 

1. Start with simple exploratory analysis: correlation matrix, distributions, and PCA/EFA to understand dimensionality.
2. If survey data: prefer EFA for interpretability; compute factor scores and feed them into models used for business KPIs.
3. If segmentation is needed and labels are unknown: perform clustering; validate clusters and translate them into business actions.
4. If you have labeled outcomes and need a classifier with interpretability: use LDA/QDA; run cross-validation and check distributional assumptions.
5. Combine approaches when needed: cluster -> validate separability with discriminant analysis -> generate operational classifiers; use EFA/PCA for feature compression.

---

If you'd like, I can also:
- Add a small diagram image (SVG) to `visualizations/` showing the conceptual map; or
- Generate a one-page slide summary or poster for stakeholder distribution.

---

Generated based on the content and case-readme files in this repository (see case folders for detailed notebooks and visualizations). For direct repo links, open `case-01-factor-analysis` (EFA), `case-02-discriminant-analysis` (LDA/QDA) and `caso-03-megamart-customer` (Clustering + PCA).

