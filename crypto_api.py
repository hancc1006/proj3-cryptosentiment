import pandas as pd
import numpy as np
import json
from collections import defaultdict
import csv
from flask import request
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

aug_btc='augmento_btc.json'
aug_eth='augmento_eth.json'
senti_btc='my_data_senti.json'

with open("./models/lr.pkl", "rb") as f:
    lr = pkl.load(f)
feature_names = lr.feature_names

def get_dates(low_range,high_range, year, month):
    date=[]
    for i in range(low_range,high_range+1):
        for j in range(0,24):
            date.append('{:04}-{:02}-{:02}_{:02}'.format(year,month,i,j))
    return date

def range_dates_full():
    days_month=[31,30,31,30,31,31,30,31,30,31]
    months=[3,4,5,6,7,8,9,10,11,12]
    dated=get_dates(13,29,2020,2)
    for i in range(len(days_month)):
        dated.append(get_dates(1,days_month[i],2020,months[i]))
    days_month_1=[31,28,31,30,31,30,31,31,30,31,30,31]
    
    months=[1,2,3,4,5,6,7,8,9,10,11,12]
    for i in range(len(days_month_1)):
        dated.append(get_dates(1,days_month_1[i],2021,months[i]))
    return dated
    


def csv_to_json(csv_file,json_file):
    csvfile=open(csv_file,'r')
    jsonfile=open(json_file,'w')
    df=pd.read_csv(csv_file)
    fieldnames=df.columns
    reader=csv.DictReader(csvfile,fieldnames)
    next(reader)
    jsonfile.write('[')
    for row in reader:
        #jsonfile.write("datas:[")
        try:
            row=list(map(int,row))
        except:
            pass
        json.dump(row,jsonfile)
        jsonfile.write(',\n')
    
    jsonfile.write(']')
    
def json_load(filename):
    r=filename
    with open(r,'r') as f:
        aug_list=json.loads(f.read())
    return aug_list

def get_mean_data():
    btc_data=json_load(aug_btc)
    eth_data=json_load(aug_eth)
    btc_senti=json_load(senti_btc)

    btc_aug=pd.DataFrame(btc_data)
    eth_aug=pd.DataFrame(eth_data)
    btc_senti_df=pd.DataFrame(btc_senti)
    #try:
    #    btc_aug1=pd.DataFrame(btc_aug['date'].str.split(' ').tolist(),columns="Date Time".split())
    #    eth_aug1=pd.DataFrame(eth_aug['date'].str.split(' ').tolist(),columns="Date Time".split())
    #except:
    #    pass
    #btc_aug[btc_aug['listing_close']=='']=0.0
    #btc_aug['twitter_negative']=btc_aug['twitter_negative'].astype('float')
    #btc_aug['twitter_optimistic']=btc_aug['twitter_optimistic'].astype('float')
    #btc_aug['bitcointalk_negative']=btc_aug['bitcointalk_negative'].astype('float')
    #btc_aug['bitcointalk_optimistic']=btc_aug['bitcointalk_optimistic'].astype('float')
    #btc_aug['reddit_optimistic']=btc_aug['reddit_optimistic'].astype('float')
    #btc_aug['reddit_negative']=btc_aug['reddit_negative'].astype('float')
    #btc_aug['listing_close']=btc_aug['listing_close'].astype('float')
    
    

    #eth_aug[eth_aug['listing_close']=='']=0.0
    #eth_aug['twitter_negative']=eth_aug['twitter_negative'].astype('float')
    #eth_aug['twitter_optimistic']=eth_aug['twitter_optimistic'].astype('float')
    #eth_aug['bitcointalk_negative']=eth_aug['bitcointalk_negative'].astype('float')
    #eth_aug['bitcointalk_optimistic']=eth_aug['bitcointalk_optimistic'].astype('float')
    #eth_aug['reddit_optimistic']=eth_aug['reddit_optimistic'].astype('float')
    #eth_aug['reddit_negative']=eth_aug['reddit_negative'].astype('float')
    #eth_aug['listing_close']=eth_aug['listing_close'].astype('float')
   
    #btc_aug['Date'], btc_aug['Time']=btc_aug1['Date'], btc_aug1['Time']
    #eth_aug['Date'], eth_aug['Time']=eth_aug1['Date'], eth_aug1['Time']
    list_features=['listing_close','twitter_negative','twitter_optimistic','bitcointalk_negative','bitcointalk_optimistic','reddit_optimistic','reddit_negative',
              'bitcointalk_positive','reddit_positive','twitter_positive','twitter_prediction','bitcointalk_prediction','reddit_prediction']
    #btc_aug1=btc_aug.groupby(['Date'])[list_features].agg(['mean']).reset_index()
    #eth_aug1=eth_aug.groupby(['Date'])[list_features].agg(['mean']).reset_index()
    return btc_aug, eth_aug, list_features

def plot_sentiment(x):
    plt.plot(x)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(x)
    plt.savefig('static/images/sentiment_plot.png')
    url1='static/images/sentiment_plot.png'
    return url1

def plot_relation(dataframe):
    s=sns.pairplot(dataframe)
    s.fig.suptitle('Pairwise Relationship Negative sentiment vs Listing price (Augmento BTC)',y=1)
    plt.savefig('static/images/pair_plot.png')
    url1='static/images/pair_plot.png'
    return url1

def plot_sample(sentiment,dataframe):
    #dataframe=dataframe.sample(500).reset_index()
    plt.plot(dataframe['listing_close'],dataframe[sentiment])
    plt.title("{} vs Listing Price Close".format(sentiment))
    plt.xlabel("Close Listing Price")
    plt.ylabel("{}".format(sentiment))
    plt.savefig('static/images/plot.png')
    url1='static/images/plot.png'
    return url1
    
def get_columns(dataframe):
    list_features=[]
    for col in dataframe.columns:
        list_features.append(col[0])
    return list_features

def get_type(dataframe):
    return type(dataframe)



def get_predictions(feature):
    x_input=[]
    for name in lr.feature_names:
        x_input_ = float(feature.get(name, 0))
        x_input.append(x_input_)
    pred_probs = lr.predict([x_input]).flat
    probs = []
    for index in np.argsort(pred_probs)[::-1]:
        prob = {
            'name': lr.target_names[index],
            'prob': round(pred_probs[index], 5)
            }
        probs.append(prob)
    return (x_input, probs)
