import csv

import pandas as pd
from flask import Flask, request, jsonify
from nltk import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import hstack
import pickle
from flask import Flask, request
import pandas as pd
import numpy as np
import Search_Engine.Search_tools as search
import json
import time
app = Flask(__name__)

@app.route('/title', methods=['GET'])
def SearchByTitle():
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    result = search.searchByTitle(query_term)
    if isinstance(result, pd.DataFrame):
        resultTranpose = result.T
        jsonResult = resultTranpose.to_json()
        return jsonResult
    else:
        json_object = {'response': '404 not found', 'similar word': result}
        return json_object

@app.route('/description', methods=['GET'])
def SearchByDescription():
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    result = search.searchByDescription(query_term)
    # check whether if result is a dataframe
    if isinstance(result, pd.DataFrame):
        resultTranpose = result.T
        jsonResult = resultTranpose.to_json()
        return jsonResult
    else:
        json_object = {'response': '404 not found', 'similar word': result}
        return json_object
if __name__ == '__main__':
    app.run(debug=True)
