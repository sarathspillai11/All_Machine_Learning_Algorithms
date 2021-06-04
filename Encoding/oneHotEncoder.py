from sklearn.preprocessing import OneHotEncoder

def oneHotEncode(dataframe):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit_transform(dataframe)
    print(dataframe.head(20))


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv(r'D:\Personal\SmartIT\data\Bakery Data Analysis\BreadBasket_DMS.csv',sep='delimiter', header=None)

    oneHotEncode(data)