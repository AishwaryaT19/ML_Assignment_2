1. Problem Statement🎓 
The objective of this assignment is to develop a predictive model to classify wine quality as "Good" (Quality score $\geq 7$) or "Average" based on 11 distinct chemical properties. This is a binary classification task performed on the Wine Quality Dataset, requiring the evaluation of six different Machine Learning algorithms to determine the most reliable predictor.


2. Model Comparison Table
The following metrics were recorded during the evaluation phase within the BITS Virtual Lab environment:
| ML Model Name       |   Accuracy |   AUC |   Precision |   Recall |   F1 |   MCC |
|:--------------------|-----------:|------:|------------:|---------:|-----:|------:|
| Logistic_Regression |       0.86 |  0.88 |        0.55 |     0.23 | 0.33 |  0.29 |
| Decision_Tree       |       0.89 |  0.77 |        0.62 |     0.6  | 0.61 |  0.54 |
| kNN                 |       0.86 |  0.79 |        0.53 |     0.21 | 0.3  |  0.27 |
| Naive_Bayes         |       0.85 |  0.86 |        0.5  |     0.79 | 0.61 |  0.55 |
| Random_Forest       |       0.9  |  0.94 |        0.71 |     0.51 | 0.59 |  0.54 |
| XGBoost             |       0.9  |  0.94 |        0.7  |     0.6  | 0.64 |  0.59 |


3. Key Observations
•	Best Performer: XGBoost is the superior model for this dataset, achieving an Accuracy of 90% and the highest MCC of 0.59, demonstrating strong reliability even with imbalanced classes.
•	Model Comparison: Ensemble methods (Random Forest and XGBoost) significantly outperformed simpler linear models like Logistic Regression.
•	Insight: While Accuracy is high across most models, the Matthews Correlation Coefficient (MCC) reveals that kNN and Logistic Regression struggle to accurately identify "Good" wines (low Recall) compared to the tree-based models.
