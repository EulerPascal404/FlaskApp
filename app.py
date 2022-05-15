###Import the necessary libraries

from flask import Flask,render_template,request,Response
import requests
import random
import tweepy
from transformers import Trainer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import time
import numpy as np
import os
import matplotlib
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

app = Flask(__name__)

###This code loads the pretrained model, as well as the tokenizer. The tokenizer will be used for preparing 
###the text, and the model is used for predicting whether a processed text is neutral, positive or negative. 
###We also set enviornment variables here. 

tokenizer = AutoTokenizer.from_pretrained("Roberta-Large")
model = AutoModelForSequenceClassification.from_pretrained("Roberta-Large", num_labels=3)
model.load_state_dict(torch.load("fold1_modelroberta-large_epoch6.pth", map_location=torch.device('cpu')))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

###This code uses the tweepy framework to find tweets about a certain topic (keyword), and eliminates retweets
###Once it has obtained 100 tweets, we terminate the search, and then add them to the raw tweets string. We finally 
###return the models predictions of raw_tweets split at the seperator. 

def TweetGenerator(keyword):
	if(keyword == ""):
		return
	client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAORicgEAAAAAy%2BK6TiiRsjrMVeGz7yTjaM%2B9R%2BM%3DVk5nXSbAU9sJ9Oyd3GnuSdDe30QHSTa0drJNCmTDwq9GP0vfPK')
	query = keyword + " stock -is:retweet"

	tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)
	raw_tweets = ""
	for tweet in tweets.data:
		raw_tweets+=tweet.text
		raw_tweets+=' ||| '
	return PredictTweet(raw_tweets.split(" ||| "))

###This function is used for generating predictions based on the tweets inputed. It also writes out a csv file containing the tweet, 
###what classification the tweet was, and how sure the model was about it. 

def PredictTweet(tweet_array):
	global counts
	counts = np.zeros(3)
	result = {
		"tweet" : [],
		"sentiment" : [],
		"confidence" : []
	}
	for tweet in tweet_array:
		conv_dict = {
			0 : "positive",
			1 : "neutral",
			2 : "negative"
		}
		tokenized_inputs = tokenizer(tweet, max_length=100, truncation=True, padding="max_length", is_split_into_words=False)
		tokenized_inputs["input_ids"] = torch.tensor(tokenized_inputs["input_ids"]).unsqueeze(0)
		tokenized_inputs["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"]).unsqueeze(0)
		with torch.no_grad():
			raw_probas = torch.nn.Softmax(dim=-1)(model(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]).logits)
		class_pred = torch.argmax(raw_probas, dim=-1).item()
		counts[class_pred] += 1
		result["tweet"].append(tweet)
		result["sentiment"].append(conv_dict[class_pred])
		result["confidence"].append(raw_probas.squeeze()[class_pred].item())
	result = pd.DataFrame(result)
	result.to_csv("static\model_response.csv", index=False)
	posT = round(100*counts[0]/len(tweet_array))
	neuT = round(100*counts[1]/len(tweet_array))
	negT = round(100*counts[2]/len(tweet_array))
	s = f"{posT}% of tweets are positive, {neuT}% of tweets are neutral, and {negT}% of tweets are negative."
	if(posT > negT and posT > neuT):
		return s + " The model predicts that this stock has a bullish sentiment surrounding it. Now might be the time to invest!"
	elif(neuT > posT and neuT > negT):
		return s + " The model predicts that the sentiment surrounding this stock is neutral. Maybe look into a Iron Condor?"
	else:
		return s + " The model predicts that this stock has a bearish sentiment surrounding it. It might be a good idea to wait for the stock to do its thing"
@app.route("/")
@app.route("/home")
def home():
	return render_template("index.html")
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
@app.route('/plot.png')
def plot_png():
	fig = create_figure()
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')
@app.route('/plot1.png')
def plot_png1():
	fig = create_figure1()
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xs = ["positive", "neutral", "negative"]
    ys = counts
    bars = ax.bar(xs, ys)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
          )
    ax.set_xlabel('Sentiments', labelpad=15, color='#333333')
    ax.set_ylabel('Frequencies', labelpad=15, color='#333333')
    ax.set_title('Frequency of Sentiments in Analyzed Tweets', pad=15, color='#333333',
         weight='bold')
    return fig
def pie_help(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
def create_figure1():
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xs = ["positive", "neutral", "negative"]
    ys = counts

    # Creating explode data
    explode = (0.0, 0.0, 0.0)

    # Creating color parameters
    colors = ( "orange", "cyan", "indigo")

    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "green" }

    wedges, texts, autotexts = ax.pie(ys,
                                  autopct = lambda pct: pie_help(pct, ys),
                                  explode = explode,
                                  labels = xs,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="magenta"))
 
    # Adding legend
    ax.legend(wedges, xs,
              title ="Frequencies",
              loc ="center left",
              bbox_to_anchor =(1, 0, 0.5, 1))

    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Sentiment Frequencies")

    return fig

@app.route("/result",methods = ['POST','GET'])
def result():
	output = request.form.to_dict()
	name = output["name"]	
	time.sleep(1)
	return render_template("index.html",name = TweetGenerator(name))

if __name__ == '__main__':
	app.run(debug = True,port = 5001)
