a. Problem Statement
The objective of this assignment is to develop a predictive model to classify wine quality as "Good" (Quality score $\geq 7$) or "Average" based on 11 distinct chemical properties. This is a binary classification task performed on the Wine Quality Dataset, requiring the evaluation of six different Machine Learning algorithms to determine the most reliable predictor.

b. Dataset Description
The dataset used is the Wine Quality Dataset, focusing on red wine samples.
•	Input Features: 11 physicochemical attributes including Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, and Alcohol.
•	Target Variable: Quality', originally an integer scale (0-10), transformed into a binary classification where scores >= 7 are labeled as 1 (Good) and scores < 7 are 0 (Average)
•	Challenge: The dataset is imbalanced, as there are significantly fewer "Good" wines than "Average" wines, making the MCC (Matthews Correlation Coefficient) a critical metric for evaluation.

2. Model Comparison Table

| ML Model Name       |   Accuracy |   AUC |   Precision |   Recall |   F1 |   MCC |
|:--------------------|-----------:|------:|------------:|---------:|-----:|------:|
| Logistic_Regression |       0.86 |  0.88 |        0.55 |     0.23 | 0.33 |  0.29 |
| Decision_Tree       |       0.89 |  0.77 |        0.62 |     0.6  | 0.61 |  0.54 |
| kNN                 |       0.86 |  0.79 |        0.53 |     0.21 | 0.3  |  0.27 |
| Naive_Bayes         |       0.85 |  0.86 |        0.5  |     0.79 | 0.61 |  0.55 |
| Random_Forest       |       0.9  |  0.94 |        0.71 |     0.51 | 0.59 |  0.54 |
| XGBoost             |       0.9  |  0.94 |        0.7  |     0.6  | 0.64 |  0.59 |

c. Observations on Performance

|index|ML Model Name|Observation about model performance|
|---|---|---|
|0|Logistic Regression|Showed high accuracy but very low recall \(0\.23\), meaning it missed many Good wines\.|
|1|Decision Tree|Balanced performance but susceptible to overfitting compared to ensemble methods\.|
|2|kNN|The weakest performer based on MCC \(0\.27\); sensitive to the scale of chemical features\.|
|3|Naive Bayes|Excellent recall \(0\.79\) but lower precision; tends to be overly optimistic about wine quality\.|
|4|Random Forest|Strong overall performance with high AUC \(0\.94\), providing very reliable predictions\.|
|5|XGBoost|Best model: achieved the highest MCC \(0\.59\) and accuracy \(0\.90\), handling the imbalanced data most effectively\.|

•	Best Performer: XGBoost is the superior model for this dataset, achieving an Accuracy of 90% and the highest MCC of 0.59, demonstrating strong reliability even with imbalanced classes.
•	Model Comparison: Ensemble methods (Random Forest and XGBoost) significantly outperformed simpler linear models like Logistic Regression.
•	Insight: While Accuracy is high across most models, the Matthews Correlation Coefficient (MCC) reveals that kNN and Logistic Regression struggle to accurately identify "Good" wines (low Recall) compared to the tree-based models.


