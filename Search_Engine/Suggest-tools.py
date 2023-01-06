import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

anime = pd.read_csv('../Resources/Jikan_database.csv')
rating = pd.read_csv('../Resources/anime_rating_1000_users.csv')
anime_features = ['mal_id', 'title', 'type', 'score', 'scored_by', 'status', 'episodes', 'aired_from', 'aired_to',
                  'source', 'members', 'favorites', 'duration', 'rating', 'nsfw', 'pending_approval', 'premiered_season',
                  'premiered_year', 'broadcast_day', 'broadcast_time', 'genres', 'themes', 'demographics', 'studios',
                  'producers', 'licensors', 'synopsis', 'background', 'main_picture', 'url', 'trailer_url', 'title_english',
                  'title_japanese', 'title_synonyms']
anime = anime[anime_features]
# print(anime['score'])
pd.options.display.max_columns = 100
#%%

merged_df = anime.merge(rating, left_on='mal_id', right_on='anime_id', how='inner')

#%%

genre_names = [
'Action', 'Adventure','Comedy', 'Drama','Sci-Fi',
'Game', 'Space', 'Music', 'Mystery', 'School', 'Fantasy',
'Horror', 'Kids', 'Sports', 'Magic', 'Romance',
]

def genre_to_category(df):
    '''Add genre category column
    '''
    d = {name :[] for name in genre_names}

    def f(row):
        genres = row.genres.split(',')
        for genre in genre_names:
            if genre in genres:
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
    # print('da anime', df.head(1).to_markdown)
    return df
def make_user_feature(df):
    # add user feature
    df['rating_count'] = df.groupby('user_id')['anime_id'].transform('count')
    df['rating_mean'] = df.groupby('user_id')['rating_y'].transform('mean')
    # print('da user', df.head(1).to_markdown)
    return df

def preprocess(merged_df):
    merged_df = make_anime_feature(merged_df)
    merged_df = make_user_feature(merged_df)
    # print(merged_df.columns)
    return merged_df

merged_df = preprocess(merged_df)
merged_df = merged_df.drop(['mal_id', 'genres'], axis=1)
# print('afterdrop:', merged_df)

merged_df = merged_df.rename(columns={'anime_id': 'mal_id'})
merged_df = merged_df.rename(columns={'rating_y': 'rating'})
print('merged_df', merged_df)
fit, blindtest = train_test_split(merged_df, test_size=0.2, random_state=0)
fit_train, fit_test = train_test_split(fit, test_size=0.3, random_state=0)

features = ['mal_id', 'title', 'type', 'score', 'scored_by', 'status', 'episodes', 'aired_from', 'aired_to',
                  'source', 'members', 'favorites', 'duration', 'rating', 'nsfw', 'pending_approval', 'premiered_season',
                  'premiered_year', 'broadcast_day', 'broadcast_time', 'themes', 'demographics', 'studios',
                  'producers', 'licensors', 'synopsis', 'background', 'main_picture', 'url', 'trailer_url', 'title_english',
                  'title_japanese', 'title_synonyms']
features += genre_names
user_col = 'user_id'
item_col = 'anime_id'
target_col = 'rating'

fit_train = fit_train.sort_values('user_id').reset_index(drop=True)
fit_test = fit_test.sort_values('user_id').reset_index(drop=True)
blindtest = blindtest.sort_values('user_id').reset_index(drop=True)
print('fit-train:', fit_train)
print('fit-test:', fit_test)

# model query data
fit_train_query = fit_train[user_col].value_counts().sort_index()
fit_test_query = fit_test[user_col].value_counts().sort_index()
blindtest_query = blindtest[user_col].value_counts().sort_index()

model = lgb.LGBMRanker(n_estimators=1000, random_state=0)
# print('features:', features)
# print('target_col:', target_col)
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

#%%

model.predict(blindtest.iloc[:10][features])

# feature importance
plt.figure(figsize=(10, 7))
df_plt = pd.DataFrame({'feature_name': features, 'feature_importance': model.feature_importances_})
df_plt.sort_values('feature_importance', ascending=False, inplace=True)
sns.barplot(x="feature_importance", y="feature_name", data=df_plt)
plt.title('feature importance')
plt.show()

def predict(user_df, top_k, anime, rating):
    user_anime_df = anime.merge(user_df, left_on='mal_id', right_on='anime_id')
    user_anime_df = make_anime_feature(user_anime_df)

    excludes_genres = list(np.array(genre_names)[np.nonzero([user_anime_df[genre_names].sum(axis=0) <= 1])[1]])

    pred_df = make_anime_feature(anime.copy())
    pred_df = pred_df.loc[pred_df[excludes_genres].sum(axis=1)==0]

    for col in user_df.columns:
        if col in features:
            pred_df[col] = user_df[col].values[0]

    preds = model.predict(pred_df[features])

    topk_idx = np.argsort(preds)[::-1][:top_k]

    recommend_df = pred_df.iloc[topk_idx].reset_index(drop=True)

    # check recommend
    print('---------- Recommend ----------')
    for i, row in recommend_df.iterrows():
        print(f'{i+1}: {row["title_japanese"]}:{row["title_english"]}')

    print('---------- Rated ----------')
    user_df = user_df.merge(anime, left_on='anime_id', right_on='mal_id', how='inner')
    for i, row in user_df.sort_values('rating',ascending=False).iterrows():
        print(f'rating:{row["rating"]}: {row["title_japanese"]}:{row["title_english"]}')

    return recommend_df

if __name__ == '__main__':
    user_id = 10
    user_df = rating.copy().loc[rating['user_id'] == user_id]
    user_df = make_user_feature(user_df)
    predict(user_df, 10, anime, rating)