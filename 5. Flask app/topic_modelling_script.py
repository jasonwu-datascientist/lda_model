# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:46:48 2018

@author: Jennifer
"""

import flask
from flask import render_template
from utilities import *

app = flask.Flask(__name__)

@app.route("/")
def hello():
    return '''
    <body>
    <h2> <center><font color = 'red'> Welcome to my topic modelling algorithm</center> </h2>
    </body>
    '''

@app.route('/page')
def page():
   with open("page.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result', methods=['POST', 'GET'])

def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form
       text = inputs['text']
       tokens = prepare_text_for_lda(text)
       prediction = predict_topic(text)
       return render_template('output.html',tokens = tokens,text=text,topic=prediction[0],
        Probability=str(prediction[1]))


       


if __name__ == '__main__':
    app.run(debug=True)