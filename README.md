# Association Rule Mining With Student Dataset

**Aim:** The aim here is to demonstrate one application of Association Rule mining with the Apriori Algorithm.

## Project Description

Association rule mining is a very important supervised machine learning method. It's used to find the relationships between different features and this in turn can be used to set association rules. An example application of association rule would be Amazon's suggestion system.

In the simplest form Amazon makes use of their users baskets data to come up with different rules stating what other items a user is likely to buy based on what she/he has already purchased.

In this case, we use the algorithm to find the rules and relationships that exists amongst a student dataset.

## Requriments

To run this project you will need to install the following installed:

1. Python 3.5 and above
2. mlxtend
3. Numpy
4. Pandas

## Project Installation

Make sure you have installed all the requirements, then follow the instruction below:

    git clone https://github.com/Emmanuel96/apriori_association_rule_mining.git
    cd apriori_association_rule_mining

## Implementation

Once our environment is set up, we import our libaries:

    # pandas, numpy ans seaborne
    import numpy as np
    import pandas as pd

    # association rule and apriori algorithm
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

### Association Rule Mining

Firstly, we read our csv file,create a data frame out of it and drop our null values as follows:

    # Please use your own directory
    data = pd.read_csv(
        r'C:/direcotry to projectapriori_association_rule_mining/Dataset/student.csv')
    student_data = pd.DataFrame(data)

Next, we drop clean our data by dropping all the null values # drop all null data
student_data.dropna(inplace=True)

Next, we convert all our featues to binary data. As we are about to carry out a binary association rule mining.

    def handle_cat_data(cat_feats, data):
        for f in cat_feats:
            to_add = pd.get_dummies(data[f], prefix=f, drop_first=True)
            merged_list = data.join(
                to_add, how='left', lsuffix='_left', rsuffix='_right')
            data = merged_list

        # then drop the categorical features
        data.drop(cat_feats, axis=1, inplace=True)
        return data

    cat_data = ['reason', 'famrel', 'freetime', 'goout', 'higher', 'Medu', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'fatherd', 'activities', 'absences', 'nursery', 'internet', 'romantic', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3',
                'nursery', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Fedu', 'Mjob', 'Fjob']
    # convert categorical and numeric features to binary data
    student_data = handle_cat_data(cat_data, student_data)

Now all we have to do is run the algorithm and print out our top rules:

    freq_item_set = apriori(student_data, min_support=0.07, use_colnames=True)

    rules = association_rules(freq_item_set, metric="lift", min_threshold=1)
    print(rules.head())

Our top 5 rules are as follows:

## License

This project is licensed under the terms of the MIT license Copyright &copy; 2020.
