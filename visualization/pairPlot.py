
#confirmed code for pairplot and distribution plot

# data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
# dataColumns = list(data.columns)
# print(list(data.columns))
# data = LabelEncoding.LabelEncode(data.values)
# data = pd.DataFrame(data,columns=dataColumns)
# sns.set_style("whitegrid")
# sns.pairplot(data, hue="AgeCategory", size=3)
# plt.savefig(outputPath+r'\diabetes_PairPlot.png')
#
# sns.FacetGrid(data,hue='Outcome',size=5).map(sns.distplot,'AgeCategory').add_legend()
# plt.savefig(outputPath+r'\diabetes_DistributionPlot.png')

# for regression

# sns.set(style="darkgrid")
#
# data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
# trainingColumns = (list(data.columns))[:-1]
# outputColumn = (list(data.columns))[-1]
# dataColumns = list(data.columns)
# print(list(data.columns))
# print('training columns : ',trainingColumns)
# print('output columns : ',outputColumn)
# data = LabelEncoding.LabelEncode(data.values)
# data = pd.DataFrame(data,columns=dataColumns,dtype='float')
# #data = sns.load_dataset(dataset)
#
# g = sns.lmplot(x="GarageArea", y=outputColumn, hue="Street", data=data)
# plt.savefig(outputPath+r'\HousePrices_Regression.png')

# for count plot
# data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
# dataColumns = list(data.columns)
# print(list(data.columns))
# data = LabelEncoding.LabelEncode(data.values)
# data = pd.DataFrame(data,columns=dataColumns)
# ax = sns.countplot(x="AgeCategory", data=data)
# plt.savefig(outputPath+r'\Diabetes_AgeCategory_CountPlot.png')

import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def pairPlotter(dataframe,outputPath,ticketId='456',hue='',size=3):

    dataframe = dataEncoding.dataEncoder(dataframe)
    sns.set_style("whitegrid")
    sns.pairplot(dataframe, hue=hue, size=size)
    plt.savefig(outputPath+r'\PairPlot_'+ticketId+'.png')

if __name__ == '__main__':
    data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
    pairPlotter(data,ticketId='4567',hue='AgeCategory',size=5)

