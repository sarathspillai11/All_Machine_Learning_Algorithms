import numpy as np
import pandas as pd
# from apyori import apriori

from LogMaster import Logger

from savemodel import saveas_sav

# data = pd.read_excel(r'C:\Users\KPMG\ELMER\QRM2017_2019_new.xlsx')
# data.replace(r'\s+', np.nan, regex=True)
#
# observations = []
# for i in range(len(data)):
#     observations.append([str(data.values[i,j]) for j in range(len(data.columns))])
#
# associations = apriori(observations, min_length = 2, min_support = 0.2, min_confidence = 0.2, min_lift = 3)
#
#
# # min_support: The minimum support of relations (float)
# # min_confidence: The minimum confidence of relations (float)
# # min_lift: The minimum lift of relations (float)
# # min_length: The minimum number of items in a rule
# # max_length: The maximum number of items in a rule
#
# associations = list(associations)
#
# print(associations[0])

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def AprioriRecomendationEngine(dataframe,transactionIdColumn, ItemsColumn,ticketId):
    log = Logger.logger
    log.info('logging started for apriori')

    hot_encoded_df = dataframe.groupby([transactionIdColumn, ItemsColumn])[ItemsColumn].count().unstack().reset_index().fillna(0).set_index(transactionIdColumn)



    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    hot_encoded_df = hot_encoded_df.applymap(encode_units)

    frequent_itemsets = apriori(hot_encoded_df, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

    saveas_sav(association_rules, 'apriori_' + ticketId + '.sav')
    return rules


