![](UTA-DataScience-Logo.png)

# Jigsaw Unintended-Bias in Toxicity Classification

* This repository holds an attempt to apply the pre-trained model BERT to news comments from the "Jigsaw Unintended Bias in Toxicity Classification" Kaggle challenge in order to detect toxicity in these news comments. 
* https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification


## Overview
  The task, as defined by the Kaggle challenge is to classify toxic news comments while making sure our model does not have any unintended bias's towards certain identities. 
  My approach in this repository will be to use Pre-trained models as well as Baye's algorithm to classify the text. The pre-trained models will be my attempt to get the best evaluation scores and Baye's algorithm will be a model that will be fast and lightweight in case of use on a smaller device. 
  Our best model was able to achieve an F1 score of 0.91.


### Data

* Data:
  * Type: The data is reprseneted in csv files which have many features however we will only focus on one column for this challenge and that is the comment_text column which displays each users comment. This is what will be our input variable. Our target or output variable will be the target column which contains the ratio of people who rated the specific comment toxic or worse. 
   
  * Size: There is about 1.21GB of data given by the challenge
  * The challenge gives us a training set of over 1 million data points and 100000 test points. However because the data is very imbalanced so I decided to downsample the data which brings the amount of training examples to about 340,000. And we will set aside 20% of that for validation.
#### Preprocessing / Clean up
* As mentioned before I decided to downsample the data so that it would be more balanced and so that my model wouldn't over train on one class. 
* I had to do some preprocessing on the comment text before I could put it into the model. This involved lemmatizing the words, removing special characters, removing stop words, and punctuation. 
* Per the challenge's request in the target column I made it so that if the value was greater than or equal to 0.5 then I would change to 1 and if it was less than 0.5 I would change it to a 0. 

#### Data Visualization
![](graph1.PNG)
![](graph2.PNG)

### Problem Formulation

* Define:
  * Input = Comment Text
  * Output = Label(either 1 or 0 for toxic or not toxic)
  * Models
    * The models I used were the pre trained BERT model using pytorch, Distilled BERT with Tensorflow, GPT2 with Tensorflow, and Naive Baye's Algorithm.
  * Loss = Binary Cross Entropy
  * Optimizer = AdamW
  * Epochs = 5
  * Batch Size = 32

### Training

* Describe the training:
  * I used Kaggle notebooks for my pytorch model and then I switched over to Google Colab when I wanted to use tensorflow this was because I wanted to use TPU's to speed up training time. 
  * For the Pytorch model each epoch took about 2 hours this was with using their gpu's as well. On Google Colab using Tensorflow and TPU's each epoch took about 15 minutes. 
  * 
  * I decided to stop training once I noticed that the validation accuracy was going down after a certain epoch.
  * Training time was a problem which was one of the reasons I swapped to Tensorflow and used TPU's.
  * Another problem was overfitting. The model's on Tensorflow had a habit of overfitting and for whatever reason the loss would have a steep decline after the first epoch as shown in the graphs. Validation accuracy would quickly plateau after the first epoch and start decreasing after more than 2 epochs. This was a problem I was struggling to solve. I had tried many variations of hyperparameters, such as changing the learning rate, changing regularization strength, number of epochs, number of training examples, and layer amount. However none seemed to fix the issue. 

### Performance Comparison
## BERT(Pytorch)

![](training_loss_graph.PNG)

![](metric_table.PNG)
![](roc_curve.PNG)
* Training Time: 8 hours
## Distilled BERT(Tensorflow)

![](dbertgraph1.PNG)
![](dfbertgraph2.PNG)
![](DbertROC.PNG)
* Training Time: 33 minutes
## GPT2(Tensorflow)

![](gpt2graph1.PNG)
![](gpt2graph2.PNG)
![](gpt2ROC.PNG)
![](gpt2eval.PNG)
* Training Time: 12 minutes
## Naïve Bayes Algorithm

![](bayesEval.PNG)
![](ROCbayes.PNG)
* Training Time: 20 seconds
### Conclusions

* Whilst using both Tensorflow and Pytorch for this project I can see that both have their merits. While pytorch allows you to customize everything from top to bottom and allows for very fine-tuned models. Tensorflow is much easier to use and allows for the use of TPU's which drastically increase training speeds. While both performed simarily the Pytorch model seemed to generalize a little better than the Tensorflow one. Baye's Algorithm is a great choice for very quick results as it only took about 20 seconds for the model to run and it outputted results extremely quick. This could be used for any device that does not want to expend great computing power to get a result that is decently accurate. GPT2 also runs significantly than BERT but sacrifices performance for it.

### Future Work

* I would like to investigate how to reduce overfitting in these pre-trained models. It could be that because the models are pre-trained they really do need very few epochs to train and thus if trained any longer we get diminishing returns.
* One thing I could look into is getting more toxic comments from various websites such as Youtube and Twitter and using those in order to get a more robust dataset.
* One thing to note is that I did not use the full dataset given to me by the kaggle challenge and there were several columns of different categorical variables. Perhaps using theese columns can allow for a better model. 

## How to reproduce results
* In order to reproduce these results you can do these in the following order:
* 1st load up the data set and clean up the text however you like feel free to use the function in the notebook
* Next downsample the data I used a 40-60 split 40 being the minority class. 
* Next Tokenize the data using the respective pre-trained model's tokenizer
* Be sure to save input ids using pickle or some other package so that when you want to re-train the model you don't have to create the ids again.
* Now you can split your ids in training and validation and initialize the model. 
* Then you fine-tune your model and wait for the results!

### Overview of files in repository

  * GPT2.ipynb: Fine-tunes the pre-trained GPT2 model and evaluates performance.(Note: This was done on Google Colab using their TPU's)
  * Bayes.ipynb: Trains Bayes model and evaluates performance.
  * DistilledBert.ipynb: This notebook contains some of the data visualization and preprocessing. It will also fine tune the pre-trained model and evaluate its performance.(Note: This was done on Google Colab using their TPU's)
  * bert-pytorch.ipynb: Fine-tunes the BERT for sequence classification model and evaluates the performance.(Note: This was done on Kaggle using their GPU's)

## Citations
* https://www.kaggle.com/code/gazu468/all-about-bert-you-need-to-know







