Analysis of Machine Learning Model Performances on detecting the difficulty level of French texts

I) Testing Logistic Regression, kNN, Decision Tree and Random Forests models :

In our project aimed at predicting the difficulty level of French texts, we evaluated four different machine learning models: Logistic Regression, kNN (k-Nearest Neighbors), Decision Tree, and Random Forests. After finding the best parameters fo each model, the performance of these models was measured in terms of precision, recall, F1-score, and accuracy. The results are displayed in the file 'final_report.pdf', which includes the Model Performances, Best Parameters for Each Model, and comparison graphics for the four models based on Precision, Recall, F1-Score, and Accuracy. We can also observe the confusions matrix for each model in the file "Model Confusion Matrices.pdf", and some examples of erroneous predictions in the file "Some erroneous predictions.pdf".

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

II) FlauBert model :

What is FlauBERT?
FlauBERT stands for "French Language Understanding Evaluation Benchmark." It is part of a family of models that adapt the original BERT architecture to specific languages, in this case, French. The model is pre-trained on a large and diverse corpus of French text, enabling it to capture a wide range of linguistic features and nuances.

The Model Used in This Project
In this project, we utilize the flaubert/flaubert_base_cased version of the FlauBERT model. This specific variant is:

- Base-sized, which means it strikes a balance between computational efficiency and model complexity, making it suitable for a variety of NLP tasks without requiring extensive computational resources.
- Cased, the model takes into account the case (uppercase or lowercase) of the letters in the text, which can be crucial for understanding the meaning and nuances in French language.

We can quickly notice that there is a huge difference between this model and the previous ones in terms of metrics and results (we can see the statistics on the file "Stats Modele Flaubert.pdf"):

Overall Performance
The model exhibits excellent performance with a uniform score of 0.92 across precision, recall, F1-score, and accuracy. This high level of accuracy indicates that the model is highly effective in classifying French text into different proficiency levels.

Confusion Matrix Insights
The model is particularly strong in identifying beginner (A1) and advanced (C1, C2) levels with high accuracy.
Some confusion is observed in intermediate levels (B1, B2), but mostly between adjacent levels (e.g., B1 misclassified as B2).

Observations from Erroneous Predictions
The errors in prediction are generally close, with misclassifications typically occurring between adjacent levels (e.g., A1 instead of A2). This suggests that while the model may occasionally confuse levels, its predictions are not drastically off.

Conclusion :

The FlauBERT model demonstrates robust performance in language proficiency classification, with its errors tending to be minor and within adjacent proficiency levels. This indicates a strong understanding of the French language nuances, though there's room for improvement in distinguishing between closely related proficiency levels.
