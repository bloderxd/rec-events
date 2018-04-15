import rec_model as rc
import calc as c


def filter_predictions_by_user(user, predictions, data_frame):
    keys = ['event', 'user']
    i1 = predictions.set_index(keys).index
    i2 = data_frame.set_index(keys).index
    recs = predictions[i1.isin(i2)]
    recs = recs.sort_values(['user', 'score'], ascending=[True, False])
    recs = recs.groupby('user').head(25)
    return recs[recs['user'] == user]


def events_by_favorite_categories(user_recs, df):
    category = df[df['event'] == user_recs.iloc[0]['event']]['category'].iloc[0]
    return df[df['category'] == category]


def fetch_popular_events_those_are_not_in_user_predictions(user_recs, recs_by_category):
    keys = ['event']
    x1 = user_recs.set_index(keys).index
    x2 = recs_by_category.set_index(keys).index
    return recs_by_category[~x2.isin(x1)]


def recommend(probably_recommendations, average):
    user_unknown_recomendation = probably_recommendations.groupby('event').score.sum() / average
    return user_unknown_recomendation.nlargest(10)


def fetch_events_for_user(user):
    predictions, data_frame = rc.build_predictions()
    recommendation = filter_predictions_by_user(user, predictions, data_frame)
    popular_events_by_user_favorite_category = events_by_favorite_categories(recommendation, data_frame)
    recommendation = fetch_popular_events_those_are_not_in_user_predictions(recommendation,
                                                                            popular_events_by_user_favorite_category)
    average = c.most_popular_event_average(recommendation)
    return recommend(recommendation, average)


if __name__ == '__main__':
    print(fetch_events_for_user(11))
