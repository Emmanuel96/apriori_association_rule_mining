# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:00:36 2020
Description: Using the Apriori algorithm to get the top 10 rules for a student dataset
@author: Emmanuel
"""


# pandas, numpy ans seaborne
import numpy as np
import pandas as pd

# association rule and apriori algorithm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# function to convert categorical data to dummy data


def handle_cat_data(cat_feats, data):
    for f in cat_feats:
        to_add = pd.get_dummies(data[f], prefix=f, drop_first=True)
        merged_list = data.join(
            to_add, how='left', lsuffix='_left', rsuffix='_right')
        data = merged_list

    # then drop the categorical features
    data.drop(cat_feats, axis=1, inplace=True)
    return data


data = pd.read_csv('dataset/student.csv')
student_data = pd.DataFrame(data)

# drop all null data
student_data.dropna(inplace=True)

cat_data = ['reason', 'famrel', 'freetime', 'goout', 'higher', 'Medu', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'fatherd', 'activities', 'absences', 'nursery', 'internet', 'romantic', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3',
            'nursery', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Fedu', 'Mjob', 'Fjob']

student_data = handle_cat_data(cat_data, student_data)

freq_item_set = apriori(student_data, min_support=0.07, use_colnames=True)

rules = association_rules(freq_item_set, metric="lift", min_threshold=1)
print(rules.head())
