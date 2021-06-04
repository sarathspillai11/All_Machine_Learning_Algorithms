from CHAID import Tree

def ChiSquaredTreeCreator(x_train,y_train,independent_variable_columns,dep_variable,dep_variable_type):

    if(dep_variable_type == 'continuous'):

        #When the dependent variable is continuous, the chi-squared test does not work due to very low frequencies of values across subgroups.
        ## create the Tree via pandas
        tree = Tree.from_pandas_df(x_train, dict(zip(independent_variable_columns, ['nominal'] * len(independent_variable_columns))), dep_variable,
                                   dep_variable_type='continuous')


    else:
        ## create the Tree via pandas
        tree = Tree.from_pandas_df(x_train, dict(zip(independent_variable_columns, ['nominal'] * len(independent_variable_columns))), dep_variable)
        ## create the same tree, but without pandas helper
        tree = Tree.from_numpy(x_train.to_array(), y_train, split_titles=independent_variable_columns, min_child_node_size=5)

    return tree
