250 word write-up:

We live in a world where the amount of data that we have available to us is growing exponentially. Our project utilizes one form of this ever-growing data: twitter tweets to help users make better and faster investing decisions about various stocks and financial items.

The first thing that the user sees is a pop-up graphic that helps them understand the model and its simplicity. Once done with that, the user can input any stock/financial item and let our backend NLP model do the heavy crunching. 

Our model analyzes tweets about the financial topic and gives the user a percent breakdown of what the general sentiment of tweets is(positive, neutral, negative). We present this to the user through graphs and show the overall sentiment on the stock, helping inform their investing decisions. Additionally,the user can download a csv file with some of the tweets that we used, what our model classified it as, and how sure our model was! We think that this last part makes the project unique as it helps the user understand the logic behind the model’s decision.

 We utilize the Roberta-Large model to make our predictions. The model is based on a pre-trained model from the Transformers library that we further trained and refined on a financial tweets dataset from Kaggle. Overall, the model gives an average accuracy of 83% for each individual tweet, but this is even higher when analyze the tweets on a certain topic in large volumes.

Note:
   The folder containing the layers of our neural network is very large(around 1.3 GB) so Github does not support the upload of that file. This file only contains the weights so that we don't have to retrain the model each time we use it. Here is the google drive link to the model: https://drive.google.com/drive/folders/1x5egep5UzpogF5WMIyg9OH0X4G5tQrqd

Kaggle Dataset that we used: https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis
