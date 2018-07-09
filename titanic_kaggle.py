# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:21:14 2018

@author: 967019
"""
# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing datasets 
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv') 
copy = train_data.copy()

train_data_survived_removed = copy.drop(['Survived'], axis = 1)
full_data = train_data_survived_removed.append(test_data)

##### Data understnding and cleaning ####
full_data.head(5)

# Colnames
list(full_data)
#12 columns are present in the dataset

# Type of columns
full_data.dtypes
# count of unique values in each column
full_data.apply(pd.Series.nunique)
full_data.info()

#### Univaiate Analysis #####
# Column Description 
# passengerID - uniuqe identifier for each passenger

# Pclass - Passenger class. 1- upper, 2- middle, 3 - lower
full_data.Pclass.value_counts().sort_values(ascending=True)
full_data.Pclass.value_counts(normalize = True) * 100
# Upper - 323 (25%)
# Middle - 277 (21%)
# Lower - 709 (54%)

# Name - Name of the passenger
full_data['Name'].head(10)

# Sex - gender
full_data.Sex.value_counts()
full_data.Sex.value_counts(normalize = True) * 100
# male - 577 (65%)
# female - 314 (35%)

# Age in numbers
sns.distplot(full_data['Age'].dropna())
full_data['Age'].describe()
# Most of passengers are aged between 20 and 40

# Number of siblings / spouses aboard the Titanic
full_data.SibSp.value_counts()
full_data.SibSp.value_counts(normalize = True) * 100
# 68% of passengers do not have any siblings and spouses
# 24% of passengers have one sibling/spouse

# Number of parents / children aboard the Titanic                            
full_data.Parch.value_counts()
full_data.Parch.value_counts(normalize = True) * 100
# 77% of passengers do not have children/parents
# 13% of passengers have 1 child/parent

# Ticket number
full_data.Ticket.head(16)
full_data.Ticket.nunique()
# 929 unique ticket numbers
# Passengers from same family might have shared the same ticket number
ticket_df = full_data.Ticket.value_counts().sort_values(ascending=False)
ticket_df = ticket_df.reset_index()
ticket_df.rename(columns ={'index': 'Ticket_id','Ticket':'No_of_passengers_per_ticket'}, inplace =True)
sns.countplot(ticket_df['No_of_passengers_per_ticket'])

# Ticket fare
full_data.Fare.head(10)
full_data.Fare.describe()
sns.distplot(full_data['Fare'].dropna())

# Cabin Number
# New variable 'Deck' can be derived from the Cabin columns

# embarked - Port of Embarkation
full_data.Embarked.value_counts()
full_data.Embarked.value_counts(normalize = True) * 100
# 70% of passengers embarked at Southampton, 21% from Cherbourg and 9% from Queenstown
                             
# Bivariate analysis
# class vs survived
g = sns.factorplot(x='Pclass', y='Survived', data=train_data, kind='bar', palette="muted", hue='Sex')
g.set_ylabels('Survival Probability')
# class , survived,  gender
    

# class, survived, age, sex
sns.factorplot(x="Pclass", y="Age", hue="Sex",
               col="Survived", data=train_data, kind="swarm")

# Creating new variable Deck from Cabin
full_data['Deck'] = full_data['Cabin'].str[0]
test_data['Deck'] = test_data['Cabin'].str[0]
train_data['Deck'] = train_data['Cabin'].str[0]
# Deck Z is for missing values
full_data['Deck'].fillna('Z', inplace = True)
test_data['Deck'].fillna('Z', inplace = True)
train_data['Deck'].fillna('Z', inplace = True)

full_data['Deck'].value_counts()

## Missing value imputation
train_data.info()
test_data.info()

g = sns.factorplot(x='Pclass', y='Survived', data=train_data, kind='bar', palette="muted",col_wrap = 3, hue='Sex', col='Deck')
## 
