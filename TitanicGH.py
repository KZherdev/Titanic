# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:12:49 2016

@author: Konstantin
Titanic Example from Kaggle
The code for data cleaning is taken from
https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic/comments

The author of model selection code is Konstantin Zherdev
"""

###################################### 1. Imports

# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #statistical data visualization
sns.set_style('whitegrid')
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

#my modification
from sklearn.model_selection import train_test_split

#####################################

##################################### 2. Download data set
# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")

# preview the data
print(titanic_df.head(3))

titanic_df.info()
print("----------------------------")
test_df.info()


######################################

###################################### 3. Clean Data
# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)


# Embarked
print(titanic_df["Embarked"].unique())
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

###################################### 4. Plot every regressor
# Embarked

sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)


# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

# Fare

# only for test_df, since there is a missing "Fare" values
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

print(avgerage_fare)
print(std_fare )
# plot
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))


avgerage_fare.index.names = std_fare.index.names = ["Survived"]
#print(avgerage_fare)
#print(std_fare )
avgerage_fare.plot(yerr=std_fare, kind='bar',legend=False)

# Age 

#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# get average, std, and number of NaN values in titanic_df

average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
#titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)
        
# plot new Age Values
#titanic_df['Age'].hist(bins=70, ax=axis2)
# test_df['Age'].hist(bins=70, ax=axis4)

# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4) # The class is used by initializing a FacetGrid object with a dataframe and the names of the variables that will form the row, column, or hue dimensions of the grid
facet.map(sns.kdeplot,'Age',shade= True) #The main approach for visualizing data on this grid is with the FacetGrid.map() method. Provide it with a plotting function and the name(s) of variable(s) in the dataframe to plot.
facet.set(xlim=(0, titanic_df['Age'].max()))#Диапазон значений для графика
facet.add_legend() # окружение, типа подппись каждого графика


# average survived passengers by age
#fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
#sns.barplot(x='Age', y='Survived', data=average_age)

# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

# plot
#fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
#sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
#sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)

# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
#sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
#sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)

# Pclass

# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
#sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

###########################################

########################################### 5. Define training and testing sets

#Minor modification - to estimate performance of our model; we are not given outcomes for test set
X = titanic_df.drop("Survived",axis=1)
Y = titanic_df["Survived"]

#X_train = titanic_df.drop("Survived",axis=1)
#Y_train = titanic_df["Survived"]
#X_test  = test_df.drop("PassengerId",axis=1).copy()

#print(X_train.describe())
#print(X_test.describe())

########################################### 6. Prediction


######################## Logistic Regression with different number of features
#Step 1: Generate additional features for the copy (polydf) of the original set
carryover = [p for p in X.columns]
polydf = X[carryover] #will consist of quadratic and intersection terms of the original X dataframe

for feature_A in carryover:
    polydf[feature_A+"^2"] = polydf[feature_A]**2
    for feature_B in carryover:
        if feature_A > feature_B:
            polydf[feature_A+"*"+feature_B] = polydf[feature_A] * polydf[feature_B]

#Step 2:Devide the polynomaial DataFrame into test and training sets
X_train, X_test, Y_train, Y_test = train_test_split(polydf, Y, test_size=0.2, random_state=0)

#Step 3: Iterate through 1 to all features
m = len(X_train.columns)
ValidError = np.empty(m, dtype = object) #to store Validation Error for every model
MisclErrorFold = np.empty(3, dtype = object) #to store misclassification error for every fold;

for i in range(2,m):
    X_train_temp = X_train.iloc[:,0:i] # take first i columns
    k_fold = KFold(n_splits=3)
    logreg = LogisticRegression()
    j = 0
    for train, test in k_fold.split(X_train_temp):
        clf = logreg.fit(X_train_temp.iloc[train], Y_train.iloc[train])
        MisclErrorFold[j] = 1-clf.score(X_train_temp.iloc[test], Y_train.iloc[test]) #неусредненный вектор для данной конкретной модели   
        j += 1
    ValidError[i] = MisclErrorFold.mean()


x = []
for i in range(m):
    x.append(i)

fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
axis1.set_title('K-fold validation error rate for logit with Polynomials, Titanic dataset')
plt.plot(x, ValidError)

axes = plt.gca()
axes.set_xlim([2,m])
axes.set_ylim([0.0,0.5])
fig = plt.gcf()
fig.set_size_inches(15, 4)
#fig.savefig("Transocean_{0}_Oil.png", dpi=100)
plt.xlabel('Complexity')
plt.ylabel('Validation Error Rate')
#plt.title("K-fold validation error rate for logit with regularization, Titanic dataset")
plt.legend()
plt.show()

#At this stage we have checked the best model size - now again estimate parameters (unfortunately, we haven't stored it anywhere) and 
#and estimate the misclassification error for this model
X_train_temp = X_train.iloc[:,0:18]
X_test_temp = X_test.iloc[:,0:18]
clf = logreg.fit(X_train_temp, Y_train)

print(clf.score(X_test_temp, Y_test))


####################### Logistic regression with regularization term
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


me = [] #misclassification error
lamb = np.arange(0.1,30,0.5)
m = len(lamb)
ValidError = np.empty(m, dtype = object) #to store Validation Error for every model
MisclErrorFold = np.empty(3, dtype = object) #to store misclassification error for every fold;

for i in range(m):
    k_fold = KFold(n_splits=3)
    logreg = LogisticRegression(C = lamb[i], penalty = "l2")
    j = 0
    for train, test in k_fold.split(X_train):
        clf = logreg.fit(X_train.iloc[train], Y_train.iloc[train])
        MisclErrorFold[j] = 1-clf.score(X_train.iloc[test], Y_train.iloc[test]) #неусредненный вектор для данной конкретной модели   
        j += 1
    ValidError[i] = MisclErrorFold.mean()

#Plot
x = [] # - x axis, length lambda, for plotting purposes
for i in range(m):
    x.append(i)
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
axis1.set_title('K-fold validation error rate for logit with regularization, Titanic dataset')

plt.plot(x, ValidError)
axes = plt.gca()
axes.set_xlim([0.0, 25])
axes.set_ylim([ValidError.min() - 0.01,ValidError.max() + 0.01])
fig = plt.gcf()
fig.set_size_inches(15, 4)
#fig.savefig("Transocean_{0}_Oil.png", dpi=100)
plt.xlabel('Regularization Parameter value')
plt.ylabel('Validation Error Rate')
#plt.title("K-fold validation error rate for logit with regularization, Titanic dataset")
plt.legend()
plt.show()

#predict using the best model (taken from Validation Error rate graph)
logreg = LogisticRegression(C = lamb[2])
clf = logreg.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))


################# KNN supervised learning
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
me = [] #misclassification error
bandwidth = np.arange(1,40,1)
m = len(bandwidth)
ValidError = np.empty(m, dtype = object) #to store Validation Error for every model
MisclErrorFold = np.empty(3, dtype = object) #to store misclassification error for every fold;

for i in range(m):
    k_fold = KFold(n_splits=3)
    knn = KNeighborsClassifier(n_neighbors = bandwidth[i], weights='uniform')
    j = 0
    for train, test in k_fold.split(X_train):
        clf = knn.fit(X_train.iloc[train], Y_train.iloc[train])
        MisclErrorFold[j] = 1-clf.score(X_train.iloc[test], Y_train.iloc[test]) #неусредненный вектор для данной конкретной модели   
        j += 1
    ValidError[i] = MisclErrorFold.mean()

#Plot
x = [] # - x axis, length lambda, for plotting purposes
for i in range(m):
    x.append(i)
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
axis1.set_title('K-fold validation error rate for KNN, Titanic dataset')

plt.plot(x, ValidError)
axes = plt.gca()
axes.set_xlim([5, m])
axes.set_ylim([ValidError.min() - 0.01,ValidError.max() + 0.01])
fig = plt.gcf()
fig.set_size_inches(15, 4)
#fig.savefig("Transocean_{0}_Oil.png", dpi=100)
plt.xlabel('Bandwidth size')
plt.ylabel('Validation Error Rate')
#plt.title("K-fold validation error rate for logit with regularization, Titanic dataset")
plt.legend()
plt.show()

#predict using the best model (taken from Validation Error rate graph)
knn = KNeighborsClassifier(n_neighbors = 13, weights='uniform')
clf = knn.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))
