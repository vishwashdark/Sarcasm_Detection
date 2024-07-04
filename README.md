# Sarcasm_Detection
Sarcasm Detection, Machine Learning, Ensemble Models, Deep Learning,  Neural Architecture, BERT, Live streaming


Sarcasm is a form of verbal mockery, which poses a unique challenge for common people to understand, this also leads to difficulties in Natural Language Processing and sentiment analysis. This study deals with Sarcasm detection from textual data and a few machine learning and ensemble models were compared with the proposed Deep Learning architecture. BERT was used for the extraction of features and vectorization of textual data. The study utilizes curated datasets consisting of non-sarcastic and sarcastic samples sourced from Kaggle and Reddit platforms. The models will be evaluated using standard metrics. The best ML model accuracy achieved was 0.88 and an F1 score of 0.89 with a combination of Bert Embeddings and the SVM model. The proposed methodology using BERT embeddings a simple neural architecture achieved an accuracy of 0.92 on the test dataset with a F1-score of 0.91. It has a far lesser complexity than other well-known Deep learning architectures in use in other works. With the live streaming of The proposed Neural Network model achieved an accuracy of 0.94 and an F measure of 0.90 on the data streamed from Reddit.



To address the issue of sarcasm detection, this study proposes a simple neural architecture model. This Neural Network model was made after extensive research done on various machine learning models, various text vectorization methods, and other famous Neural Network models such as the LSTM model. Different Feature extractions like TF-IDF, Word2Vec, and BERT were performed on the preprocessed data, as seen in Fig. 1., which was then fed into the ML models, SVM, KNN, Random Forest, AdaBoost, XGBoost, Logistic Regression, and an Ensemble Classifier. BERT was fed into an LSTM Deep learning model and the proposed Deep Learning model also for comparison. The proposed Neural Model’s architecture, as seen in Fig. 2, will be discussed later.

![image](https://github.com/vishwashdark/Sarcasm_Detection/assets/92641662/f1e4d561-8495-4d0e-9a4e-ea4fb380dba3)

A.	Data Preprocessing:
 As a part of processing data, the data needs to be pre-processed into TensorFlow vectors first. The input layer is converted to tokens of length 1x16, and the choice of 16 token sequences is done to increase the computational efficiency during the process of BERT vectorization. 
 
B.	Feature Extraction : 
A feature extraction step follows the preprocessing stage. Here the tokens undergo 3 different feature extractions, as seen in Fig.1, TF-IDF, Word2Vec and BERT Embeddings. These resulting vectors are then fed to Machine Learning models and BERT embeddings alone is fed to LSTM and the proposed neural model. The BERT embeddings are present as tokens, which are further used to train the proposed simple neural network model. The model uses an encoder-decoder network where it takes the text as input and returns vectors as the output. 

C.	Machine Learning Models:
After transforming the preprocessed data using diverse embeddings, including TF-IDF, Word2Vec, and BERT, we employed a comprehensive suite of ML models to tackle the task of sarcasm detection. The array of models encompassed “Support Vector Machines”, “K-Nearest Neighbors”, “Random Forest”, “Logistic Regression”, “GDBboost”, “ADAboost”, and an “Ensemble Classifier”. Each model played a pivotal role in training and testing the sarcasm detection framework.

 Notably, the ensemble classifier was skillfully crafted by leveraging the strengths of “Support Vector Machines” and “Logistic Regression”, thereby enhancing the overall predictive performance and robustness of the sarcasm detection system. This holistic approach allowed for a thorough exploration of diverse model architectures and their collective impact on the accuracy and reliability of sarcasm detection in natural language data.
 
D.	Proposed Neural Network model:
The BERT Embeddings are then fed into a Neural Architecture consisting of 4 layers as shown in Fig. 2.:
•	Lambda: This layer is used to extract a pooled output from the BERT layer which gives a shape of 1x768
•	Dense: This layer uses ReLU, which converts the output to 1x128
•	Dropout: This layer is used to reduce the vector size to the required size where we use a 0.2 dropout rate.
•	Dense: Finally, here, the value from the last layer is then processed to predict the class of the input.

The final output from the last layer helps predict the model.

The proposed simple neural network architecture made using BERT Embeddings is designed to process 16 token TensorFlow input by integrating BERT Embeddings and custom Neural Network Layers, this model aims to capture contextual information for the detection of Sarcasm.

![image](https://github.com/vishwashdark/Sarcasm_Detection/assets/92641662/b17dc577-ca13-493e-bf6b-bea7f21b4fbf)


Dataset Description

Two datasets were used for this work, the first dataset on Sarcasm detection is sourced from Twitter , it contains 3 columns: the article link, its Headline, and the column is_sarcastic to show the output, the Headline consists of the Sarcastic text and the is_sarcastic is the label given to the heading which is 0 and 1, where 0 stands for not sarcastic and 1 stand for sarcastic. The dataset consists of a total of 26,709 articles and their corresponding labels. The dataset consists of 11,724 sarcastic articles and 14,985 not sarcastic articles. 

The second dataset used for experiments is streamed from Reddit and consists of 13629 records of which all are sarcastic statements. The Data streaming is done from Reddit using PRAW (Python Reddit API wrapper) PRAW which helps simplify the process of making requests to the Reddit API. Using PRAW we used keywords related to sarcasm such as 'Sarcasm', 'Irony', 'hyperbole', 'Disingenuousness', 'Snarky', 'Heavysigh', 'facepalm', 'Skepticism', 'sarcasm' and collected around 16000 sarcasm posts from Reddit which were sarcasm texts. On the other hand, for non-sarcastic text we used common topics such as "Informative" "Helpful" "Interesting" "Serious" "Genuine" "Sincere" "Authentic" "Insightful" and "Thought-provoking" and collected around 15300 posts.
