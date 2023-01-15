import csv
import string

import nltk
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import ranker
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from Search_Engine.Search_tools import remove_puncts
from rank_bm25 import BM25Okapi

# remove prefix suffix (not cover all suffix)
def removeSuffix(list):
    newList = []
    for word in list:
        if word.endswith('fullness'):
            word = word[:-len('fullness')]
        if word.endswith('lessness'):
            word = word[:-len('lessness')]
        if word.endswith('less'):
            word = word[:-len('less')]
        if word.endswith('ness'):
            word = word[:-len('less')]
        if word.endswith('ly'):
            word = word[:-len('ly')]
        if word.endswith('tic'):
            word = word[:-len('tic')]
        if word.endswith('ing'):
            word = word[:-len('ing')]
        newList.append(word)
        print(newList)
    return newList
pd.options.display.max_columns = 100
anime = pd.read_csv('../Resources/Jikan_database_with_keyword.csv', encoding = "ISO-8859-1")

title = anime['synopsis']
corpus = title.to_numpy().tolist()
cleaned_corpus = []
for doc in corpus:
    cleaned_doc = remove_puncts(doc, string)
    cleaned_corpus.append(cleaned_doc)
tokenized_clean_corpus = []
for doc in cleaned_corpus:
    doc = doc.split()
    cleaned_doc = removeSuffix(cleaned_doc)
    tokenized_clean_corpus.append(doc)
bm25 = BM25Okapi(tokenized_clean_corpus)
# print('bm25', bm25.get_scores('military').tolist())
user1000_rating = pd.read_csv('../Resources/anime_rating_1000_users.csv')
user_anime_lookup_rating = pd.DataFrame({'user_id': [0, 0, 0], 'anime_id': [25, 41, 653], 'rating': [8, 7, 10]})
anime_features = ['mal_id', 'title', 'type', 'score', 'scored_by', 'status', 'episodes', 'aired_from', 'aired_to',
                  'source', 'members', 'favorites', 'duration', 'rating', 'nsfw', 'pending_approval', 'premiered_season',
                  'premiered_year', 'broadcast_day', 'broadcast_time', 'genres', 'themes', 'demographics', 'studios',
                  'producers', 'licensors', 'synopsis', 'background', 'main_picture', 'url', 'trailer_url', 'title_english',
                  'title_japanese', 'title_synonyms', 'synopsis_keyword']
genre_names = [
    'Action', 'Adventure', 'Avant Garde', 'Award Wining', 'Boys Love', 'Comedy', 'Drama',
    'Fantasy', 'Girls Love', 'Gourmet', 'Horror', 'Mystery', 'Romance',
    'Sports', 'Supernatural', 'Sci-Fi', 'Slice of Life', 'Suspense'
]
demographic_names = [
    'Josei', 'Kids', 'Seinen', 'Shoujo', 'Shounen'
]
theme_names = [
    'Adult Cast', 'Anthropomorphic', 'CGDCT', 'Childcare', 'Combat Sports',
    'Crossdressing', 'Delinquents', 'Detective', 'Educational', 'Gag Humor',
    'Gore', 'Harem', 'High Stakes Game', 'Historical', 'Idols (Female)',
    'Idols (Male)', 'Isekai', 'Iyashikei', 'Love Polygon', 'Magical Sex Shift',
    'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music',
    'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts',
    'Pets', 'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romantic Subtext',
    'Samurai', 'School', 'Showbiz', 'Space', 'Strategy Game', 'Super Power', 'Survival', 'Team Sports',
    'Time Travel', 'Vampire', 'Video Game', 'Visual Arts', 'Workplace'
]
lower_case_theme_names = list(map(lambda x: x.lower(), theme_names))
lower_case_genre_names = list(map(lambda x: x.lower(), genre_names))
lower_case_demographic_names = list(map(lambda x: x.lower(), demographic_names))
words = set(nltk.corpus.words.words())
keywordScore = pd.read_csv('../Resources/top55_BM25score.csv')
# print(list(keywordScore['word']))
anime = anime[anime_features]
def average(lst):
    return sum(lst) / len(lst)

# Create a csv that replicated to jikan database, but adding keyword column
# def synopsisToList(df):
#      corpus = df['synopsis'].to_numpy().tolist()
#      cleaned_corpus = []
#      for doc in corpus:
#          cleaned_doc = remove_puncts(doc, string)
#          cleaned_doc = cleaned_doc.split()
#          cleaned_doc = removeSuffix(cleaned_doc)
#          cleaned_doc = list(set(cleaned_doc).intersection(set(list(keywordScore['word']))))
#          cleaned_corpus.append(cleaned_doc)
#      df['synopsis_keyword'] = cleaned_corpus
#      df.to_csv('../Resources/Jikan_database_with_keyword.csv', index=False)
#      return df
# anime = synopsisToList(anime)
#%%
# merge 2 dataframes
max_id = user_anime_lookup_rating["user_id"].max()
def generate_new_id(id):
  return id + max_id + 1
user1000_rating["user_id"] = user1000_rating["user_id"].apply(generate_new_id)
rating = pd.concat([user_anime_lookup_rating, user1000_rating], join="outer", ignore_index=True)
rating.reset_index(inplace=True)
rating.drop(columns=["index"], inplace=True)
# set a new id to 'user1000_rating' dataframe
merged_df = anime.merge(rating, left_on='mal_id', right_on='anime_id', how='inner')
# print(merged_df)

## Create a csv that store all word with avg bm25 score attach to it.
# remove suffix and word that have length less than 4
# description_file = open("../Resources/description.txt")
# textual_synopsis = description_file.read()
# textual_synopsis = textual_synopsis.split()
# # Prevent the feature overlap
# # Remove prefix suffix
# textual_synopsis = removeSuffix(textual_synopsis)
# textual_synopsis = list(((set(textual_synopsis) - set(lower_case_theme_names)) - set(lower_case_genre_names)) - set(lower_case_demographic_names))
#
# textual_synopsis = [i for i in textual_synopsis if i in words]
# textual_synopsis = [i for i in textual_synopsis if len(i) > 3]
# print(textual_synopsis)
# scoreList = []
#
# # find average bm25 score
# for word in textual_synopsis:
#     scores = bm25.get_scores(word).tolist()
#     if scores != []:
#         scoresMean = round(average(scores), 6)
#         scoreList.append(scoresMean)
#         print(scoresMean)


keywordWithScore = pd.read_csv('../Resources/top55_BM25score.csv')
keywordList = keywordWithScore[['word', 'round_score']].values.tolist()
keyword_word_list = list(keywordWithScore['word'])
def keyword_to_category(df):
    '''Add keyword category column
    '''
    d = {name :[] for name in keyword_word_list}
    # print(type(keywordList))
    def f(row):
        keywords = row.synopsis_keyword.split(',')
        for i, keyword in enumerate(keyword_word_list):
            if keyword in keywords[0]:
                d[keyword].append(keywordList[i][1])
            else:
                d[keyword].append(0)

    # create synopsis category dict
    df.apply(f, axis=1)

    # add genre category
    keyword_df = pd.DataFrame(d, columns=keyword_word_list)
    df = pd.concat([df, keyword_df], axis=1)
    # print(df)
    return df
def theme_to_category(df):
    '''Add theme category column
    '''
    d = {name :[] for name in theme_names}
    # print(d)
    def f(row):
        themes = row.themes.split(',')
        for theme in theme_names:
            if theme in themes[0]:
                d[theme].append(1)
            else:
                d[theme].append(0)
    # create theme category dict
    df.apply(f, axis=1)

    # add genre category
    theme_df = pd.DataFrame(d, columns=theme_names)
    df = pd.concat([df, theme_df], axis=1)
    return df
def demographic_to_category(df):
    '''Add genre category column
    '''
    d = {name :[] for name in demographic_names}
    def f(row):
        demographics = row.demographics.split(',')
        for demographic in demographic_names:
            if demographic in demographics[0]:
                d[demographic].append(1)
            else:
                d[demographic].append(0)
    # create genre category dict
    df.apply(f, axis=1)

    # add genre category
    demographic_df = pd.DataFrame(d, columns=demographic_names)
    df = pd.concat([df, demographic_df], axis=1)
    return df
def genre_to_category(df):
    '''Add genre category column
    '''
    d = {name :[] for name in genre_names}
    def f(row):
        genres = row.genres.split(',')
        for genre in genre_names:
            if genre in genres[0]:
                d[genre].append(1)
            else:
                d[genre].append(0)
    # create genre category dict
    df.apply(f, axis=1)

    # add genre category
    genre_df = pd.DataFrame(d, columns=genre_names)
    df = pd.concat([df, genre_df], axis=1)
    return df

def make_anime_feature(df):
    # convert object to a numeric type, replacing Unknown with nan.
    df['score'] = df['score'].apply(lambda x: np.nan if x=='Unknown' else float(x))
    # add genre category columns
    df = genre_to_category(df)
    df = demographic_to_category(df)
    df = theme_to_category(df)
    df = keyword_to_category(df)
    return df
def make_user_feature(df):
    # add user feature
    df['rating_count'] = df.groupby('user_id')['anime_id'].transform('count')
    df['rating_mean'] = df.groupby('user_id')['rating_y'].transform('mean')
    return df

def preprocess(merged_df):
    merged_df = make_anime_feature(merged_df)
    merged_df = make_user_feature(merged_df)
    return merged_df

merged_df = preprocess(merged_df)
# print('merged_df', merged_df)
merged_df = merged_df.drop(['mal_id', 'genres', 'demographics', 'themes', 'synopsis', 'synopsis_keyword'], axis=1)

merged_df = merged_df.rename(columns={'anime_id': 'mal_id'})
merged_df = merged_df.rename(columns={'rating_y': 'rating'})
fit, blindtest = train_test_split(merged_df, test_size=0.2, random_state=0)
fit_train, fit_test = train_test_split(fit, test_size=0.3, random_state=0)

features = ['score', 'scored_by', 'members', 'favorites', 'rating_count', 'rating_mean', 'nsfw']
features += genre_names
features += demographic_names
features += theme_names
features += keyword_word_list
user_col = 'user_id'
item_col = 'anime_id'
target_col = 'rating'

fit_train = fit_train.sort_values('user_id').reset_index(drop=True)
fit_test = fit_test.sort_values('user_id').reset_index(drop=True)
blindtest = blindtest.sort_values('user_id').reset_index(drop=True)

# model query data
fit_train_query = fit_train[user_col].value_counts().sort_index()
fit_test_query = fit_test[user_col].value_counts().sort_index()
blindtest_query = blindtest[user_col].value_counts().sort_index()

model = lgb.LGBMRanker(n_estimators=1000, random_state=0)
model.fit(
    fit_train[features],
    fit_train[target_col],
    group=fit_train_query,
    eval_set=[(fit_test[features], fit_test[target_col])],
    eval_group=[list(fit_test_query)],
    eval_at=[1, 3, 5, 10], # calc validation ndcg@1,3,5,10
    early_stopping_rounds=100,
    verbose=10
)

model.predict(blindtest.iloc[:10][features])
# display SciView
plt.figure(figsize=(10, 7))
df_plt = pd.DataFrame({'feature_name': features, 'feature_importance': model.feature_importances_})
df_plt.sort_values('feature_importance', ascending=False, inplace=True)
sns.barplot(x="feature_importance", y="feature_name", data=df_plt)
plt.title('feature importance')
plt.show()
def predict(user_df, top_k, anime, rating):
    user_anime_df = anime.merge(user_df, left_on='mal_id', right_on='anime_id')
    user_anime_df = make_anime_feature(user_anime_df)
    # print(user_anime_df)
    excludes_genres = list(np.array(genre_names)[np.nonzero([user_anime_df[genre_names].sum(axis=0) <= 1])[1]])
    excludes_demographics = list(np.array(demographic_names)[np.nonzero([user_anime_df[demographic_names].sum(axis=0) <= 1])[1]])
    excludes_themes = list(np.array(theme_names)[np.nonzero([user_anime_df[theme_names].sum(axis=0) <= 1])[1]])
    excludes_synopsis = list(np.array(keyword_word_list)[np.nonzero([user_anime_df[keyword_word_list].sum(axis=0) <= 1])[1]])
    pred_df = make_anime_feature(anime.copy())
    pred_df = pred_df.loc[pred_df[excludes_genres].sum(axis=1)==0]
    pred_df = pred_df.loc[pred_df[excludes_demographics].sum(axis=1) == 0]
    pred_df = pred_df.loc[pred_df[excludes_themes].sum(axis=1) == 0]
    pred_df = pred_df.loc[pred_df[excludes_synopsis].sum(axis=1) == 0]
    # drop an anime if that user is already add to favorite
    ids_to_drop = user_anime_df['mal_id'].tolist()
    pred_df = pred_df[~pred_df['mal_id'].isin(ids_to_drop)]

    for col in user_df.columns:
        if col in features:
            pred_df[col] = user_df[col].values[0]
    preds = model.predict(pred_df[features])
    topk_idx = np.argsort(preds)[::-1][:top_k]

    recommend_df = pred_df.iloc[topk_idx].reset_index(drop=True)

    return recommend_df

