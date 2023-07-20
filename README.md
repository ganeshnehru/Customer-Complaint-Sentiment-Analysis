---
title: Comsumer Complaints Classification
date: "November 30, 2022"
---

# Introduction

This project aims to construct classification models for consumer complaints using text mining and natural language processing techniques. This analysis will provide valuable insights into consumer behavior, supporting businesses in implementing effective strategies to enhance customer satisfaction and retention.

The solution involves a two-fold approach. Firstly, standard sentiment analysis techniques will be applied to classify complaints into predefined categories. Secondly, topic modeling using LDA (Latent Dirichlet Allocation) will be utilized to discover potential new categories and gain deeper insights into customer concerns.

The implementation and evaluation of LSTM, SVM, MNB, and KNN models will be conducted, with each model offering unique strengths for a comprehensive comparison and selection of the most suitable approach.

# Data Exploration and Preprocessing

### Feature Identification:
Initial observation revealed that the dataset contained eighteen feature columns. However, most features were redundant for the scenario. Therefore, only a few features were collected from the consumers while submitting a complaint, such as: 
* Product
* Consumer complaint narrative
* Issues
* Sub-issues (not including personal information)

Complaint narratives were provided by consumers around 55% of the time, while predefined issues were listed in the form for the rest. The "Submitted-via" feature recorded the medium used by consumers to file complaints, with seven unique values: web, email, referral, phone, web referral, fax, and mail post. Complaints from different mediums exhibited different volumes of complaints about different products, suggesting that using the "Submitted-via" feature alongside the "consumer complaint narrative" could boost the F1 score and classification accuracy.


### Complaint Narrative Analysis:
The analysis addressed whether the model could learn the input and target features. Complaint narratives were preprocessed by removing stop words and symbols. Frequent word counts were extracted for complaint narratives of different product types. Word clouds for the top 20 frequent words per product type illustrated significant differences, indicating that the data was learnable and learning was feasible.

# Methods

### Long Short Term Memory (LSTM)

The LSTM network was built using the NLTK and Keras library. Complaint narratives were tokenized and padded to a maximum sequence length of 50. The dataset was split into train and test sets with an 80-20 split, and 10% of the test set was used for validation during training. The LSTM model's training time was reduced using CuDNNLSTM. The model was trained for 10 epochs with the default settings of the Adam optimizer. However, the validation loss did not improve significantly after 2 epochs, indicating potential overfitting.

The trained model's performance was evaluated on the test dataset, yielding an F1 score of 0.86 and a classification accuracy of 86%. However, category 6 ("Crypto Currency") exhibited poor evaluation metrics due to insufficient training samples, making learning infeasible for this category.

### Multinomial Naive Bayes (MNB)

During preprocessing, redundant features were removed, resulting in a new data frame containing product and customer complaint narratives. Similar product titles were addressed by relabeling and merging them, reducing the overall number of products from eighteen to seven. Null values were dropped from the dataset, resulting in a condensed dataset used for modeling.

After preprocessing, a new data frame was created from the preprocessed complaint file. The complaint narratives with null values were dropped, and the remaining data frame consisted of 1112339 datapoints. The products were encoded into numeric values, and the training and test sets were instantiated. TF-IDF vectorizer was used to measure word relevancy in the complaint narratives. Using the Multinomial Naïve Bayes classifier, a prediction accuracy of 82% was achieved.

### Support Vector Machine (SVM)

Before training SVM and Logistic Regression models, the training dataset was obtained after processing the 'narrative' column for the dataset. As a pre-processing technique, stopwords were removed from each entry, including strings like 'xx/xx/xxxx', punctuations, and numericals to eliminate noise in the training data. Due to SVM's slow performance with large datasets, 100,000 samples were randomly selected from the dataset. The training data was split into 80% train and 20% test for validation purposes. 

Text vectorization was performed using TF-IDF transformation on the training and test set. Singular Value Decomposition was performed with 250 components to enhance SVM model performance. Since SVM requires normalized values, the training and test dataset were scaled using StandardScaler. The SVC was trained with C=1.0 and probability estimates set to True.

Validation was performed on the test set, and the F1 scoring metric was used to determine model performance. The SVM model obtained a score of 0.87099 on the validation set.

### K-Nearest Neighbors (KNN)
To begin with the KNN implementation, pre-processing steps were needed on the text data. A count vectorizer was built to remove stop-words, clean, and tokenize the textual data. TF-IDF transformation was performed after cutting down the dataset to only the product and complaint narratives of interest.

Once the pre-processing was done and the TF-IDF transformation was complete, the product categories were encoded into numerical values. A KNNClassifier model was built on top of this data, limiting the sample size to 100,000 for standardization across implementations.

The performance of each model was evaluated using the macro-averaged precision, recall, and F1 scores. The LSTM model displayed better recall, F1, and accuracy compared to other models, with KNN showing the highest precision. LSTM was selected as the best-performing model for the classification problem, considering all metrics.

# Comparisons

| Method      | Precision | Recall | F1 Score | Accuracy |
| :---        |    ----   |   ---- |   ---    | ---      |
| LSTM        | 0.72      | 0.85   | 0.73     | 0.86     |
| SVD         | 0.72      | 0.70   | 0.71     | 0.86     |
| Naive Bayes | 0.72      | 0.69   | 0.70     | 0.85     |
| KNN         | 0.73      | 0.72   | 0.70     | 0.72     |

# Conclusions

The models built in this project achieved an accuracy of 86% and streamlined the consumer complaint filing process. Target variable ambiguity was mitigated by renaming certain categories based on data exploration. However, due to insufficient data, classifying complaint narratives of type "Crypto Currency" proved challenging.

Future work will focus on improving classification accuracy through novel methods and additional features. Features like "Submitted-via" were identified during data exploration to influence the target variable, suggesting potential enhancements in model performance.