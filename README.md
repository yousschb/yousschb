Analysis of Machine Learning Model Performances on detecting the difficulty level of French texts

In our project aimed at predicting the difficulty level of French texts, we evaluated four different machine learning models: Logistic Regression, kNN (k-Nearest Neighbors), Decision Tree, and Random Forests. The performance of these models was measured in terms of precision, recall, F1-score, and accuracy. Here is a detailed analysis of the results:

Logistic Regression
Precision: 0.403
Recall: 0.409
F1-Score: 0.402
Accuracy: 0.409
Logistic Regression showed moderate performance with a slight preference for recall over precision. This indicates that the model is slightly better at correctly identifying positive classes but at the cost of increased false positives.

k-Nearest Neighbors (kNN)
Precision: 0.401
Recall: 0.292
F1-Score: 0.240
Accuracy: 0.292
The kNN model exhibited lower performance, particularly in terms of recall and F1-score. This suggests that the model struggles to correctly identify positive classes, which could be due to an imbalance in class distribution or poor neighbor selection.

Decision Tree
Precision: 0.289
Recall: 0.294
F1-Score: 0.289
Accuracy: 0.294
The Decision Tree showed the weakest performance among the tested models. With relatively low scores across all metrics, this might indicate overfitting to the training data or an inability to capture the complexity of the data.

Random Forests
Precision: 0.372
Recall: 0.379
F1-Score: 0.359
Accuracy: 0.379
Random Forests displayed slightly better performance than Logistic Regression and significantly better than kNN and Decision Tree. This model seems to offer a better balance between precision and recall, suggesting better generalization compared to the other models.

Conclusion
Considering all metrics, Random Forests appear to be the most effective model for this specific task, although the overall scores suggest there is still room for improvement. The relatively low performance of the models could be due to several factors, such as data quality, the need for more advanced preprocessing techniques, or the necessity to explore other models or more sophisticated neural network architectures.

It's also important to note that while accuracy is a useful metric, it should not be the sole criterion for judgment, especially in cases where there is a significant class imbalance. Metrics such as the F1-score, which combines precision and recall, can offer a more balanced perspective on model performance.
