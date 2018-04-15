from itertools import groupby as g


def most_common(iterator):
    return max(g(sorted(iterator)), key=lambda (x, v): (len(list(v)), -iterator.index(x)))[0]


def most_popular_event_average(probably_recommendations):
    return probably_recommendations[
        probably_recommendations['event'] == most_common(probably_recommendations.values.tolist())[1]
        ].count()[0]
