from flask import Flask, request, jsonify, make_response
from flask import Flask, request
import pandas as pd
import Search_Engine.Search_tools as search
import Search_Engine.Suggest_tools as suggest
app = Flask(__name__)

@app.route('/title', methods=['GET'])
def SearchByTitle():
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    result = search.searchByTitle(query_term)
    # check whether if result is a dataframe
    if isinstance(result, pd.DataFrame):
        resultTranpose = result.T
        jsonResult = resultTranpose.to_json()
        response = make_response(jsonResult)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    else:
        jsonResult = {'response': '404', 'similar': result}
        response = make_response(jsonResult)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

@app.route('/description', methods=['GET'])
def SearchByDescription():
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    result = search.searchByDescription(query_term)
    # check whether if result is a dataframe
    if isinstance(result, pd.DataFrame):
        resultTranpose = result.T
        jsonResult = resultTranpose.to_json()
        response = make_response(jsonResult)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    else:
        jsonResult = {'response': '404', 'similar': result}
        response = make_response(jsonResult)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

@app.route('/suggestion', methods=['GET'])
def Suggestion():
    argList = request.args.to_dict(flat=False)
    query_term = int(argList['query'][0])
    anime = suggest.anime
    rating = suggest.rating
    user_df = rating.copy().loc[rating['user_id'] == query_term]
    user_df = user_df.rename(columns={'rating': 'rating_y'})
    user_df = suggest.make_user_feature(user_df)
    recommend_df = suggest.predict(user_df, 40, anime, rating)
    resultTranpose = recommend_df[['mal_id', 'title', 'main_picture']].T
    jsonResult = resultTranpose.to_json()
    response = make_response(jsonResult)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

if __name__ == '__main__':
    app.run(debug=True)
