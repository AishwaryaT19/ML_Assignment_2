🎯 Project Overview
This project uses Machine Learning to classify wine quality as 'Good' (Score $\geq 7$) or 'Average' based on 11 chemical features. The goal was to deploy a live Streamlit application that evaluates 6 different models.


| ML Model Name       |   Accuracy |   AUC |   Precision |   Recall |   F1 |   MCC |
|:--------------------|-----------:|------:|------------:|---------:|-----:|------:|
| Logistic_Regression |       0.86 |  0.88 |        0.55 |     0.23 | 0.33 |  0.29 |
| Decision_Tree       |       0.89 |  0.77 |        0.62 |     0.6  | 0.61 |  0.54 |
| kNN                 |       0.86 |  0.79 |        0.53 |     0.21 | 0.3  |  0.27 |
| Naive_Bayes         |       0.85 |  0.86 |        0.5  |     0.79 | 0.61 |  0.55 |
| Random_Forest       |       0.9  |  0.94 |        0.71 |     0.51 | 0.59 |  0.54 |
| XGBoost             |       0.9  |  0.94 |        0.7  |     0.6  | 0.64 |  0.59 |

Observations for your README:

Best Model: XGBoost performed the best with an Accuracy of 0.90 and an MCC of 0.59.

Worst Model: kNN struggled with this dataset, likely due to the high dimensionality of the chemical features.

Insight: Ensemble methods (Random Forest/XGBoost) significantly outperformed simple linear models.
