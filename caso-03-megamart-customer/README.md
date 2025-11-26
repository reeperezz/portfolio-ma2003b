# MegaMart Customer Segmentation ReadMe 
**(Business Context)**

## Client & problem description:

MegaMart is a large retail company with thousands of customers and no formal segmentation strategy. Marketing efforts are broad and inefficient, resulting in low engagement, inconsistent conversion rates, and high churn among inactive customers.
This project aims to identify natural behavioral customer segments and translate them into actionable marketing strategies.

## Strategic importance:

Understanding customer segments empowers MegaMart to:

Personalize communication and promotions

Reduce churn among low-activity customers

Increase conversions among high-browsing shoppers

Protect revenue from high-value loyal customers

Allocate marketing resources efficiently with measurable ROI

Segmentation also helps define targeted campaigns, improve customer lifetime value, and support CRM personalization.

**(Methodology)**

## Multivariate methods applied:

Hierarchical Clustering (Ward linkage) to explore natural group structures and guide the choice of k.

K-Means Clustering (final model with k = 4) for stable, interpretable segment formation.

PCA (Principal Component Analysis) for visualization of clusters in 2D and dimensionality inspection.

## Justification of the choice:

K-Means is ideal for standardized behavioral data and gives clear, actionable customer groups.

Hierarchical clustering confirms cluster structure and helps determine the optimal number of clusters.

PCA provides an intuitive visualization of high-dimensional groups.

## Tools and libraries used:

Python:
pandas, numpy, scikit-learn, scipy

Visualization:
matplotlib, seaborn

Execution:
Jupyter Notebook (retail.ipynb)

Utility functions:
utils.py (scaling, clustering, PCA, and plotting helpers)

(Data)

## Dataset description:

File: retail_customer_data.csv

3,000 customers with 9 behavioral features describing purchasing, engagement, browsing, recency, and lifecycle activity.

## Key variables:

Purchase behavior:
monthly_transactions, avg_basket_size, total_spend

Engagement:
avg_session_duration, product_views_per_visit, email_open_rate

Lifecycle:
customer_tenure_months, recency_days

Returns:
return_rate


## (Main Findings)

Optimal number of clusters: 4, validated through hierarchical clustering, elbow method, and silhouette scores.

Final K-Means solution (k = 4) shows the following distribution:

Cluster 0 (17.5%) — High-Value Loyalists
High spend (~$6507), high engagement, consistent behavior.

Cluster 1 (31%) — At-Risk Minimal Shoppers
Very low spend (~$422), long inactivity, minimal engagement.

Cluster 2 (14.4%) — Established Steady Buyers
High basket size (~18), strong spend (~$3875), long tenure.

Cluster 3 (37.1%) — Browsers with Moderate Spend
High product views, moderate spend (~$1450), selective buyers.

PCA visualization explains 61.98% of total variance (PC1 = 41%, PC2 = 21%).

Silhouette score of final model: ~0.317, good for real-world behavioral data.

**(Featured Visualizations)**

Las visualizaciones están en la carpeta /visualizations.

Customer Distributions

Correlation Matrix

Dendrograms (Comparison)

Ward Detailed Dendrogram

Elbow Plot

Elbow + Silhouette Analysis

Silhouette Plot

Cluster Visualization (PCA)

Cluster Profiles Heatmap

**(Model performance metrics)**

Silhouette Score: ~0.317

Cluster Stability: consistent assignments across random seeds

Cluster Cohesion: strongest cohesion in Clusters 0 and 3

PCA Separation: clear separation across PC1 and PC2

Cluster means (behavioral profiles) are available in the notebook.

**(Business recommendations)**

Retain High-Value Loyalists (Cluster 0):

VIP benefits, early-access promotions, personalized appreciation

Financial impact: protects ≈ $104K/year

Reactivate At-Risk Shoppers (Cluster 1):

Simple reactivation reminders, seasonal offers

Impact: recovers ≈ $19K/year

Convert Browsers into Buyers (Cluster 3):

Browse-triggered offers, product bundles

Impact: adds ≈ $161K/year

Support Established Steady Buyers (Cluster 2):

Upsell opportunities based on basket patterns

Reinforce long-term engagement

**(Expected impact)**

Short term (1–3 months):
Activation campaigns for Cluster 1; loyalty perks for Cluster 0.

Medium term (3–9 months):
Conversion tests for Cluster 3; curated recommendations for Cluster 2.

Long term (>9 months):
Integrate segmentation into CRM; monitor cluster transitions; refine personalization.

**(Next steps)**

Add product-category features for more granular segmentation.

Build a churn prediction model using clusters as predictive features.

Track customer movement between clusters over time (behavioral drift).