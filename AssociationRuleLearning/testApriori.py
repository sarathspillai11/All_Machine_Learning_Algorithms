import pandas as pd
import Master.MasterAlgorithm as master
from collections import Counter

if __name__ == '__main__':
    user_info = pd.read_csv(r'D:\Personal\takehome\training.tsv',
                            delimiter='\t', encoding='utf-8', names=['user_id', 'Date', 'activity'])
    combination = user_info.groupby("user_id")['activity'].apply(lambda activity: ','.join(activity)).to_frame()
    combination["EmailOpen"] = 0
    combination["FormSubmit"] = 0
    combination["Purchase"] = 0
    combination["EmailClickthrough"] = 0
    combination["CustomerSupport"] = 0
    combination["PageView"] = 0
    combination["WebVisit"] = 0
    for i, row in combination.iterrows():
        for key in Counter(row['activity'].split(',')).keys():
            ifor_val = Counter(row['activity'].split(','))[key]
            combination.at[i, key] = ifor_val
        if row['Purchase'] > 0:
            combination.at[i, 'Purchase'] = 1

    x = combination.loc[:, ['EmailOpen', 'FormSubmit', 'EmailClickthrough', 'CustomerSupport', 'PageView', 'WebVisit']]
    y = combination.loc[:, 'Purchase']
    print(combination.columns)
    print(combination.head(5))

    data =  master.findCombination(dataframe=combination,inputType = 'labelled',mlType='supervised',contentType = 'text',usecaseType = 'recomendation',encodingType = 'label',custom = 'apriori')
    #data.to_excel(r'D:\Personal\SmartIT\data\MARS House Prices\Predicted_EM.xlsx')
    print(data)