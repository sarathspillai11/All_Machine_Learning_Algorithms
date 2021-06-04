from StatisticalTests.ChiSquaredTest import chi_squared_test


import pandas as pd
import numpy  as np
from copy import deepcopy
from savemodel import saveas_sav
from scipy import stats



# Function: Prediction
def prediction_dt_id3(model, Xdata):
    print('inside func :', Xdata.columns)
    Xdata = Xdata.reset_index(drop=True)
    ydata = pd.DataFrame(index=range(0, Xdata.shape[0]), columns=["predicted"])
    data = pd.concat([ydata, Xdata], axis=1)
    rule = []
    print('inside func after reset index :', Xdata.columns)

    # Preprocessing - Boolean Values
    for j in range(0, data.shape[1]):
        if data.iloc[:, j].dtype == "bool":
            data.iloc[:, j] = data.iloc[:, j].astype(str)

    # Preprocessing - Binary Values
    for j in range(1, data.shape[1]):
        if data.iloc[:, j].dropna().value_counts().index.isin([0, 1]).all():
            for i in range(0, data.shape[0]):
                if data.iloc[i, j] == 0:
                    data.iloc[i, j] = "0"
                else:
                    data.iloc[i, j] = "1"

    data = data.applymap(str)
    dt_model = deepcopy(model)

    for i in range(0, len(dt_model)):
        dt_model[i] = dt_model[i].replace("{", "")
        dt_model[i] = dt_model[i].replace("}", "")
        dt_model[i] = dt_model[i].replace(".", "")
        dt_model[i] = dt_model[i].replace("IF ", "")
        dt_model[i] = dt_model[i].replace("AND", "")
        dt_model[i] = dt_model[i].replace("THEN", "")
        dt_model[i] = dt_model[i].replace("=", "")

    for i in range(0, len(dt_model) - 2):
        splited_rule = [x for x in dt_model[i].split(" ") if x]
        rule.append(splited_rule)

    # print('########### Rule ########### :',rule)

    for i in range(0, Xdata.shape[0]):
        for j in range(0, len(rule)):
            rule_confirmation = len(rule[j]) / 2 - 1
            rule_count = 0
            for k in range(0, len(rule[j]) - 2, 2):
                print('@@@@@@@@@@ :', data.columns)
                if (data[rule[j][k]][i] in rule[j][k + 1]):
                    rule_count = rule_count + 1
                    if (rule_count == rule_confirmation):
                        data.iloc[i, 0] = rule[j][len(rule[j]) - 1]
                else:
                    k = len(rule[j])

    for i in range(0, Xdata.shape[0]):
        if data.iloc[i, 0] == "nan":
            data.iloc[i, 0] = dt_model[len(dt_model) - 1]

    return data


# Function: ID3 Algorithm
def dt_id3(Xdata, ydata, pre_pruning="none", chi_lim=0.1,ticketId=''):
    ################     Part 1 - Preprocessing    #############################
    # Preprocessing - Creating Dataframe


    name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    dataset = pd.concat([ydata, Xdata], axis=1)

    # Preprocessing - Boolean Values
    for j in range(0, dataset.shape[1]):
        if dataset.iloc[:, j].dtype == "bool":
            dataset.iloc[:, j] = dataset.iloc[:, j].astype(str)

    # Preprocessing - Binary Values
    for j in range(0, dataset.shape[1]):
        if dataset.iloc[:, j].dropna().value_counts().index.isin([0, 1]).all():
            for i in range(0, dataset.shape[0]):
                if dataset.iloc[i, j] == 0:
                    dataset.iloc[i, j] = "0"
                else:
                    dataset.iloc[i, j] = "1"

    dataset = dataset.applymap(str)

    # Preprocessing - Unique Words List
    unique = []
    uniqueWords = []
    for j in range(0, dataset.shape[1]):
        for i in range(0, dataset.shape[0]):
            token = dataset.iloc[i, j]
            if not token in unique:
                unique.append(token)
        uniqueWords.append(unique)
        unique = []

        # Preprocessing - Label Matrix
    label = np.array(uniqueWords[0])
    label = label.reshape(1, len(uniqueWords[0]))

    ################    Part 2 - Initialization    #############################
    # ID3 - Initializing Variables
    i = 0
    branch = [None] * 1
    branch[0] = dataset
    gain = np.empty([1, branch[i].shape[1]])
    rule = [None] * 1
    rule[0] = "IF "
    root_index = 0
    skip_update = False
    stop = 2

    ################     Part 3 - id3 Algorithm    #############################
    # ID3 - Algorithm
    while (i < stop):
        entropy = 0
        denominator_1 = branch[i].shape[0]
        for entp in range(0, label.shape[1]):
            numerator = (branch[i][(branch[i].iloc[:, 0] == label[0, entp])].count())[0]
            if numerator > 0:
                entropy = entropy - (numerator / denominator_1) * np.log2((numerator / denominator_1))
        gain.fill(entropy)
        for element in range(1, branch[i].shape[1]):
            if len(branch[i]) == 0:
                skip_update = True
                break
            if len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1 or gain.sum(axis=1) == 0:
                rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + "."
                rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                break
            if i > 0 and pre_pruning == "chi_2" and chi_squared_test(branch[i].iloc[:, 0],
                                                                     branch[i].iloc[:, element]) > chi_lim:
                if "." not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x: x.value_counts().index[0])[
                        0] + "."
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                continue
            for word in range(0, len(uniqueWords[element])):
                denominator_2 = (branch[i][(branch[i].iloc[:, element] == uniqueWords[element][word])].count())[0]
                for lbl in range(0, label.shape[1]):
                    numerator = (branch[i][(branch[i].iloc[:, 0] == label[0, lbl]) & (
                            branch[i].iloc[:, element] == uniqueWords[element][word])].count())[0]
                    if numerator > 0:
                        gain[0, element] = gain[0, element] + (denominator_2 / denominator_1) * (
                                numerator / denominator_2) * np.log2((numerator / denominator_2))

        if skip_update == False:
            gain[0, 0] = -gain[0, 0]
            root_index = np.argmax(gain)
            gain[0, 0] = -gain[0, 0]
            rule[i] = rule[i] + list(branch[i])[root_index]

            for word in range(0, len(uniqueWords[root_index])):
                branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])
                rule.append(rule[i] + " = " + "{" + uniqueWords[root_index][word] + "}")

            for logic_connection in range(1, len(rule)):
                if len(np.unique(branch[i][0])) != 1 and rule[logic_connection].endswith(" AND ") == False and rule[
                    logic_connection].endswith("}") == True:
                    rule[logic_connection] = rule[logic_connection] + " AND "
        skip_update = False
        i = i + 1
        # print("iteration: ", i)
        stop = len(rule)

    for i in range(len(rule) - 1, -1, -1):
        if rule[i].endswith(".") == False:
            del rule[i]

    rule.append("Total Number of Rules: " + str(len(rule)))
    rule.append(dataset.agg(lambda x: x.value_counts().index[0])[0])
    print("End of Iterations")
    saveas_sav(dt_id3, 'id3_' + ticketId + '.sav')
    return rule

    ############### End of Function ##############


# if __name__ == '__main__':
#     from sklearn.model_selection import train_test_split
#     ##=============================================================================
#     df = pd.read_csv(r'D:\Personal\SmartIT\test files\ID3_Input.csv', sep=';')
#
#     X = df.iloc[0:6, 0:4]
#     y = df.iloc[0:6, 4]
#
#     print('Xdata type :', type(X))
#     print('ydata type :', type(y))
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
#     dt_model = dt_id3(Xdata=X_train, ydata=y_train, pre_pruning="chi_2", chi_lim=0.1)
#
#     # Prediction
#     test = df.iloc[6:14, 0:4]
#     print('test  :',test)
#     out = prediction_dt_id3(dt_model, X_test)
#     print(out)
#     out.to_excel(r'D:\Personal\SmartIT\data\sampleID3Out.xlsx')
#     #=============================================================================




