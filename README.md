# Playground

This depository stores some of my experimented projects. Welcome to check it out and give any feedback, so that we can improve the cases even better.

**[01-Classification-Modeling-on-Hotel-Scoring](https://github.com/TomLin/Playground/blob/master/01-Classification-Modeling-on-Hotel-Scoring.ipynb)**<br/>
Comparing the performance of models among MLP, Xgboost, and Logistic, and choose **Logistic Model** as the final choice. **The final model reaches 0.62 in accuracy and 0.61 in F1 score in three classes scenario.** If we look deeper, the share of each class are 35%, 29%, 36%, which compared to the precision from the predicted result for each class 68%, 47%, 65%, we can say that **Logistic Model doubles the precision** generally. 

**[02-Mass-Transit-Analysis](https://github.com/TomLin/Playground/blob/master/02-Mass-Transit-Analysis.ipynb)**<br/>
By the end of this analysis, I am able to come up with a list of suggested stations for ads placement. Adhering  to the stations, I also obtain a table of expected commute time for various subway routes.


**[04-Model-Comparison-Word2vec-Doc2vec-TfIdfWeighted](https://github.com/TomLin/Playground/blob/master/04-Model-Comparison-Word2vec-Doc2vec-TfIdfWeighted.ipynb)**<br/>
In this post, I'd like to test out the effect of different kinds of wordembeddings on text classifier. 
The wordembeddings investigated here include word2vec, TF-IDF weighted word2vec, pre-train GloVe word2vec and doc2vec.
It turns out that there is no significant difference among their performance, in the meantime, pre-train Glove and doc2vec alone seem to under-perform a bit compared with others.   

**[05-Try-out-Spacy-Pretrain](https://github.com/TomLin/Playground/blob/master/05-Try-out-Spacy-Pretrain.ipynb)**<br/>
This post focuses on trying out spaCy's new BERT-style language pre-training feature.
First, I set up a classifier without pre-training and then compare it with one initiated with pre-training.
It turns out the classifier without pre-training has higher score on both f1 and accuracy.
Nonetheless, take it with a grain of salt. If we look closely, the one with pre-training performs better at identifying rare classes.
It still has its own merits if identifying rare classes matter.

**[06-Recsys-Variants-SVD-Recommender](https://github.com/TomLin/Playground/blob/master/06-1-Recsys-Data-Preprocess.ipynb)**<br/>
This post consists of 5 jupyter notebookes, beginning from data preprocessing to model training. In this post, we focus on CF, particularly SVD-based algorithms. We'll practice on building not only SVD based model, but also its variations, such as one with regularization and the other, neural network adopted. The final founding is that recommender built on neural network(NCF) has the best performance in terms of RMSE. Nevertheless, most SVD based models can roughly achieve quality result as long as each item and user of the dataset having enough interaction records, meaning no significant problem of cold-start.
 



