from flask import Flask,render_template,request
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/greet",methods=["POST"])
def greet():

    gold_data=pd.read_csv("./static/gld_price_data.csv")
    print(gold_data)

    """**SUMMARIZE THE DATASET**"""
    # print first 5 rows in the dataframe
    gold_data.head()

    # print last 5 rows in the dataframe
    gold_data.tail()

    #getting the statistical measures of the data
    gold_data.describe()

    correlation = gold_data.corr()

    '''constructing a heatmap to understand the correlatiom
    plt.figure(figsize = (8,8))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')'''

    # correlation values of GLD
    print(correlation['GLD'])

    # checking the distribution of the GLD Price
    #sns.distplot(gold_data['GLD'],color='green')

    X = gold_data.drop(['Date','GLD'],axis=1)
    Y = gold_data['GLD']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
    regressor = RandomForestRegressor(n_estimators=100)

    # training the model
    regressor.fit(X_train,Y_train)

    # prediction on Test Data
    #test_data_prediction = regressor.predict(X_test)

    # prediction on Test Data

    spx = request.form.get("spx")
    usd = request.form.get("usd")
    slv = request.form.get("slv")
    esu = request.form.get("esu")


    """**PREDICTING /TESTING THE DATA**"""
    inp=[spx,usd,slv,esu]
    test_data = regressor.predict([inp])
    
    return render_template("greet.html",result=test_data)

    

