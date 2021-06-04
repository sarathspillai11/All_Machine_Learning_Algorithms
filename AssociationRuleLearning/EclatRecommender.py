from LogMaster import Logger
from savemodel import saveas_sav
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


def EclatRecommendationEngine(dataframe,transactionIdColumn, ItemsColumn,ticketId):
    log = Logger.logger
    log.info('logging started for eclat')


    hot_encoded_df = dataframe.groupby([transactionIdColumn, ItemsColumn])[ItemsColumn].count().unstack().reset_index().fillna(0).set_index(transactionIdColumn)

    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    hot_encoded_df = hot_encoded_df.applymap(encode_units)
    res = fpgrowth(hot_encoded_df, min_support=0, use_colnames=True)
    res = association_rules(res, metric="lift", min_threshold=1)
    saveas_sav(association_rules, 'eclat_' + ticketId + '.sav')

    return res

