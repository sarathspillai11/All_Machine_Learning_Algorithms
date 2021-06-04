

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd

def countVectorize(dataframe):

    vect = CountVectorizer()


    # use TreeBankWordTokenizer
    tokenizer = TreebankWordTokenizer()
    vect.set_params(tokenizer=tokenizer.tokenize)

    # remove English stop words
    vect.set_params(stop_words='english')

    # include 1-grams and 2-grams
    vect.set_params(ngram_range=(1, 2))

    # ignore terms that appear in more than 50% of the documents
    vect.set_params(max_df=0.5)

    # only keep terms that appear in at least 2 documents
    vect.set_params(min_df=2)


    sentenceDf = pd.DataFrame({idx:[row[0]] for idx,row in dataframe.iterrows()})
    doc_vec = vect.fit_transform(sentenceDf.iloc[0])

    print('feature names :', vect.get_feature_names())

    # Create dataFrame
    vectorisedDf = pd.DataFrame(doc_vec.toarray().transpose(),
                       index=vect.get_feature_names())

    # Change column headers
    vectorisedDf.columns = sentenceDf.columns

    print(vectorisedDf)

    x_train = doc_vec.toarray()

    return x_train


if __name__ == '__main__':
    textData = pd.read_excel(r'D:\Personal\SmartIT\test files\regression input.xlsx')
    countVectorize(textData)


