Analysis of Machine Learning Model Performances on detecting the difficulty level of French texts

In our project aimed at predicting the difficulty level of French texts, we evaluated four different machine learning models: Logistic Regression, kNN (k-Nearest Neighbors), Decision Tree, and Random Forests. After finding the best parameters fo each model, the performance of these models was measured in terms of precision, recall, F1-score, and accuracy. The results are displayed in the file 'final_report.pdf', which includes the Model Performances, Best Parameters for Each Model, and comparison graphics for the four models based on Precision, Recall, F1-Score, and Accuracy. We can also observe the confusions matrix for each model in the file "Model Confusion Matrices.pdf".

Based on the results, we can observe that :

Logistic Regression
- Challenges with Intermediate Levels: Logistic Regression showed particular difficulty in accurately predicting the intermediate levels A2, B1, and B2. This suggests a challenge in distinguishing between these nuanced levels of language proficiency.
- Better at Extreme Levels: The model appears to perform relatively better at the extreme levels (beginner and advanced), but struggles with the gradations in between.

kNN
- Struggles with Specific Levels: The kNN model had notable difficulties, especially with levels B1 and C2. A significant observation is the model's tendency to overwhelmingly predict many instances as C1, including those that are actually C2, indicating a bias towards this particular level.
- Misclassification Across the Board: There is a general trend of misclassification across various levels, with the model showing a lack of precision in distinguishing between the nuanced differences in language proficiency.
  
Decision Tree
- Weakest Performance at Intermediate Levels: The Decision Tree exhibited its weakest performance with intermediate levels, showing a wide disparity. Many levels were predicted, but with a considerable number of errors, indicating a lack of consistency and reliability in the model's predictions.
- Generalization Issues: The spread of errors across different levels suggests issues with generalization, possibly due to overfitting or an inability to capture the subtleties of language proficiency levels.
  
Random Forests
- Highest Overall Performance: Random Forests demonstrated the highest overall performance among the models. However, it still showed notable errors, particularly in predicting levels A2 and B1.
- Better Balance but Not Perfect: While this model offered a better balance in precision and recall compared to the others, it still faced challenges in accurately classifying certain levels, indicating room for improvement in distinguishing between closely related proficiency levels.
  
Conclusion :

Considering all metrics and the insights from the confusion matrices, Random Forests appear to be the most effective model for this specific task of predicting language proficiency levels. However, the detailed analysis reveals that even the best-performing model has its challenges, particularly with certain intermediate levels like A2 and B1. This suggests that while Random Forests offer a better balance in precision and recall, there is still considerable room for improvement.

The relatively lower performance of the models, especially in distinguishing between intermediate levels, could be attributed to several factors. These include the complexity of the data, the need for more advanced preprocessing techniques, and possibly the requirement to explore other, more sophisticated neural network architectures that might capture the nuances of language proficiency more effectively.

It's also crucial to consider that while accuracy is a valuable metric, it should not be the sole criterion for judgment. This is particularly important in scenarios where there might be a significant class imbalance. Metrics like the F1-score, which combines precision and recall, provide a more balanced perspective on model performance. They are particularly useful in highlighting how well a model can manage the trade-off between correctly identifying as many instances of a particular class as possible (recall) and ensuring that those identifications are accurate (precision).
