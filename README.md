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

III) Attempts to Improve the Code for French Text Difficulty Prediction

In my project aimed at predicting the difficulty level of French texts, I undertook a series of steps to refine and improve the model's performance. Here's an overview of the key stages in this process:

1. Initial Testing with DistilBERT Multi-Lingual Model
Initially, I employed the DistilBERT model in its multi-lingual variant. DistilBERT is known for its efficiency and effectiveness in natural language processing tasks, offering a lighter and faster alternative to the full-sized BERT models. The multi-lingual version was chosen for its broad language coverage, including French.

2. Implementing Cross-Validation and Optimal Model Selection
To enhance the model's robustness and generalizability, I integrated a 5-fold cross-validation technique. This approach helped in assessing the model's performance across different data segments. Additionally, I focused on selecting the best-performing model configuration from these cross-validation runs, aiming to find the most effective model settings.

3. Transition to CamemBERT for French Language Focus
Confronting issues of overfitting, I shifted to using CamemBERT, a model specifically trained on French language data. This decision was based on the hypothesis that a language-specific model like CamemBERT would be more adept at capturing the intricacies and nuances of the French language, compared to a multi-lingual model.

4. Advanced Strategies to Combat Overfitting
To further tackle the overfitting challenge, I employed more sophisticated cross-validation methods, coupled with early stopping mechanisms to prevent the model from excessively learning from the training data. I also introduced batch normalization to stabilize the learning process. Adjusting the learning rate was another critical step, fine-tuning the training dynamics to optimize the model's performance.

5. Data Cleaning and Stop Word Removal
In an effort to streamline the dataset, I embarked on a data cleaning process. This involved removing stop words and unnecessary characters. The rationale behind this was to reduce noise in the data and focus the model's learning on more relevant aspects of the text. However, this approach did not yield the expected results. I realized that this method of data cleaning was counterproductive, as it led to the loss of crucial contextual information. The removal of stop words and certain characters inadvertently stripped the texts of elements that were significant for accurately determining their difficulty levels.

6. Experimenting with data augmentation through paraphrasing
Seeking to enhance the dataset, I explored data augmentation by paraphrasing the sentences. This involved translating the French sentences into English and then back into French, using a translation tool (TextBlob). The idea was to expand the dataset with variations of the original sentences, thereby providing the model with a richer set of training data. However, this approach had an unintended consequence. The paraphrasing process altered the complexity of the sentences, sometimes simplifying or complicating them in ways that did not accurately reflect their original difficulty levels.

7. Discovery of FlauBERT and its precision
As I delved deeper into the BERT library, my journey led me to a pivotal discovery: FlauBERT. This model, specifically designed for the French language, stood out for its remarkable precision in handling tasks related to French text analysis. FlauBERT, a variant of the well-known BERT model but fine-tuned for French, offered a level of specificity and accuracy that was not attainable with the multi-lingual models I had previously used.

8. Switching to a french-specific model
Realizing the potential of a language-specific approach, I decided to pivot from the multi-lingual model to using FlauBERT, which was exclusively trained on French data. This shift was a game-changer. The French-specific model was inherently more adept at understanding the nuances and complexities of the French language, leading to significantly improved predictions in text difficulty levels.

9. Employing TFFlaubertModel and TFFlaubertForSequenceClassification
To fully harness the capabilities of FlauBERT, I utilized TFFlaubertModel and TFFlaubertForSequenceClassification. These TensorFlow implementations of FlauBERT allowed me to leverage the model's strengths in a more flexible and efficient manner. TFFlaubertModel provided the foundational architecture, capturing the intricacies of French syntax and semantics, while TFFlaubertForSequenceClassification added a layer specifically tailored for classifying the difficulty levels of French texts.

10. Refinement and results
With these tools at my disposal, I was able to fine-tune the model to a high degree of accuracy. The combination of FlauBERT's language-specific focus and the tailored classification layer led to a model that not only understood the subtleties of French text but also accurately gauged its complexity relative to language learners' proficiency levels. The results were promising, showing a marked improvement over my initial attempts with multi-lingual models. This journey through various models and strategies underscored the importance of choosing the right tools and approaches for specific language processing tasks, especially when dealing with the intricacies of language learning and proficiency assessment.

11. Final Adjustments and Achieving Optimal Results

In the final stage of refining my model, a crucial breakthrough came from adjusting key parameters, specifically the number of epochs and the batch size. By carefully tuning these parameters, I was able to significantly enhance the model's performance.

Adjusting the number of epochs allowed the model to learn from the data more thoroughly without overfitting. Finding the right balance was key; too few epochs meant underfitting, while too many led to the model picking up too much noise from the training data. Similarly, tweaking the batch size helped in managing the computational load and the granularity of the learning process. A smaller batch size often led to more stable and reliable gradient updates, but required more iterations, whereas a larger batch size provided faster computations but with less stable updates.

These adjustments proved to be highly effective. The model's accuracy improved remarkably, achieving scores of 0.598 and 0.599, a significant leap from the initial baseline of 0.540 with my basic models. This improvement underscored the importance of not only choosing the right model and approach but also fine-tuning the training process to suit the specific characteristics of the task at hand. It was a testament to the nuanced nature of machine learning, where small changes in parameters can lead to substantial improvements in performance.

IV) Streamlit App

