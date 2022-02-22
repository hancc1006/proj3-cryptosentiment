from flask import Flask
from flask import request, render_template, redirect
import os
from crypto_api import *
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.datastructures import MultiDict
app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))

@app.route('/', methods=["GET", "POST"])  # the site to route to, index/main in this case
def get_data():
    btc_aug, eth_aug, list_features=get_mean_data()
    type=get_type(btc_aug)
    #btc_col=get_columns(btc_aug)
    #eth_col=get_columns(eth_aug)
    x_input, predictions = get_predictions(request.args)
    if request.method=='POST':
        #print(MultiDict(request.form['sentiment']))
        dataframe=request.form['dataframe']
        sentiment=request.form['sentiment']
        if dataframe=='btc_aug':
            dataframe=btc_aug
            type=get_type(dataframe)
            x_input, predictions = get_predictions(request.args)
            plt.clf()
            plt.scatter(btc_aug['listing_close'],btc_aug[sentiment],alpha=0.2)
            plt.title("{} vs Listing Price Close for Augmento BTC".format(sentiment))
            plt.xlabel("Close Listing Price")
            plt.ylabel("{}".format(sentiment))
            plt.grid()
            plt.savefig('static/images/plot.jpg')
            url='static/images/plot.jpg'
            return render_template('crypto.html',btc_aug=btc_aug,btc_col=list_features, eth_aug=eth_aug,url=url,feature_names=feature_names, x_input=x_input, prediction=predictions)
        if dataframe=='eth_aug':
            dataframe=btc_aug
            type=get_type(dataframe)
            x_input, predictions = get_predictions(request.args)
            plt.clf()
            plt.scatter(eth_aug['listing_close'],eth_aug[sentiment], alpha=0.2)
            plt.title("{} vs Listing Price Close for Augmento ETH".format(sentiment))
            plt.xlabel("Close Listing Price")
            plt.ylabel("{}".format(sentiment))
            plt.grid()
            plt.savefig('static/images/plot.jpg')
            url='static/images/plot.jpg'
            return render_template('crypto.html',btc_aug=btc_aug,btc_col=list_features, eth_aug=eth_aug,url=url,feature_names=feature_names, x_input=x_input, prediction=predictions)
    
    return render_template('crypto.html',btc_aug=btc_aug,btc_col=list_features, eth_aug=eth_aug,type=type, feature_names=feature_names,x_input=x_input, prediction=predictions)



if __name__ == '__main__':
    #diagnose any errors you come across when running the code
    #debug=True
    app.run()

