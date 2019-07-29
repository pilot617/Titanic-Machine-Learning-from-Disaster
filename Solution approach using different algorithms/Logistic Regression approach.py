# classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# modules to handle data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later


train = pd.read_csv(r'C:\Users\Student\Python Codes\Titanic Project\train.csv')
test = pd.read_csv(r'C:\Users\Student\Python Codes\Titanic Project\test.csv')
##print(train)
##print(test_df)
# save PassengerId for final submission
passengerId = test.PassengerId

# merge train and test
titanic = train.append(test, ignore_index=True, sort=True)
# create indexes to separate data later on
train_idx = len(train)
test_idx = len(titanic) - len(test)
titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
# map the normalized titles to the current titles 
titanic.Title = titanic.Title.map(normalized_titles)

# group by Sex, Pclass, and Title 
grouped = titanic.groupby(['Sex','Pclass', 'Title'])

# apply the grouped median value on the Age NaN
titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

# fill Cabin NaN with U for unknown
titanic.Cabin = titanic.Cabin.fillna('U')
# find most frequent Embarked value and store in variable
most_embarked = titanic.Embarked.value_counts().index[0]

# fill NaN with most_embarked value
titanic.Embarked = titanic.Embarked.fillna(most_embarked)
# fill NaN with median fare
titanic.Fare = titanic.Fare.fillna(titanic.Fare.median())

# size of families (including the passenger)
titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1

# map first letter of cabin to itself
titanic.Cabin = titanic.Cabin.map(lambda x: x[0])

# Convert the male and female groups to integer form
titanic.Sex = titanic.Sex.map({"male": 0, "female":1})
# create dummy variables for categorical features
pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")
title_dummies = pd.get_dummies(titanic.Title, prefix="Title")
cabin_dummies = pd.get_dummies(titanic.Cabin, prefix="Cabin")
embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")
# concatenate dummy columns with main dataset
titanic_dummies = pd.concat([titanic, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)

# drop categorical fields
titanic_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# create train and test data
train = titanic_dummies[ :train_idx]
test = titanic_dummies[test_idx: ]

# convert Survived back to int
train.Survived = train.Survived.astype(int)
# create X and y for data and target values 
X = train.drop('Survived', axis=1).values 
y = train.Survived.values
# create array for test set
X_test = test.drop('Survived', axis=1).values

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))
def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)
def fit(x, y, theta):
    opt_weights = opt.fmin_tnc(func=cost_function, x0=theta,
                  fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]
parameters = fit(X, y, theta)
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
p = probability(parameters, X_test)
Survived =  np.zeros((len(p),), dtype=np.int)
for i in range(len(p)):
    if(p[i]>=0.5):
        Survived[i] = 1
    else:
        Survived[i] = 0
#Survived = np.asarray(Survived)
# dataframe with predictions
kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': Survived})
# save to csv
kaggle.to_csv(r'C:\Users\Student\Python Codes\Titanic Project\titanic_pred.csv', index=False)

print(kaggle.shape)


# view value counts for the normalized titles
##print(titanic.Title.value_counts())
#print(titanic_dummies.head())
