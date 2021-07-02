import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns
#Create your df here:

df = pd.read_csv("~/Desktop/Coding/capstone_starter/profiles.csv")


#print(df["status"].value_counts())
#print(df["body_type"].value_counts())
#print(df["religion"].value_counts())
#I am going to start by sorting out and cleaning my qualitative data


# I narrowed it down the groups to average, fit, curvy, skinny. I had to group most of the curvy and skinny people together since average takes up most of the sample.
body_mapping = { "athletic": "fit",
"thin": "skinny", "a little extra": "curvy",
 "full figured": "curvy", "overweight": "curvy",
"jacked": "fit", "used up": "skinny", "rather not say": "curvy"}


# highschool = 0, 2 year college = 1, undergrad = 2, masters = 3, phd = 4, med =5, law = 6, (working or dropouts or other)= NA
education_mapping = {"graduated from college/university": 2,
"graduated from masters program": 3,
"working on college/university": np.nan,
"working on masters program": np.nan,
"graduated from two-year college": 1,
"graduated from high school": 0,
"graduated from ph.d program": 4,
"graduated from law school": 6,
"working on two-year college": np.nan,
"dropped out of college/university": np.nan,
"working on ph.d program": np.nan,
"college/university": 2,
"graduated from space camp":  np.nan,
"dropped out of space camp":  np.nan,
"graduated from med school":  5,
"working on space camp":  np.nan,
"working on law school":  np.nan ,
"two-year college":  1,
"working on med school ":   np.nan,
"dropped out of two-year college":   np.nan,
"dropped out of masters program ":   np.nan,
"masters program":   3,
"dropped out of ph.d program":  np.nan,
"dropped out of high school":  np.nan,
"high school":   0,
"working on high school":    np.nan,
"space camp":     np.nan,
"ph.d program":    4,
"law school":      6,
"dropped out of law school":    np.nan,
"dropped out of med school":    np.nan,
"med school":    5}



#relgion diets at 0, vegan at 1, vegetarian at 2, anything at 3, and other at 4

diet_mapping = {'halal': 0,
                'strictly halal': 0,
                'mostly halal': 0,
                'kosher': 0,
                'strictly kosher': 0,
                'mostly kosher': 0,
                'vegan': 1,
                'strictly vegan': 1,
                'mostly vegan': 1,
                'vegetarian': 2,
                'strictly vegetarian': 2,
                'mostly vegetarian': 2,
                'anything': 3,
                'strictly anything': 3,
                'mostly anything': 3,
                'other': 4,
                'strictly other': 4,
                'mostly other': 4}



df["education"] = df["education"].map(education_mapping)
df["body_type"] = df["body_type"].map(body_mapping)
df["diet"] = df["diet"].map(diet_mapping)
df["religion"] = df["religion"].str.split().str.get(0)


df= df.dropna(subset = ["education","drinks","drugs","diet","smokes", "body_type","religion"])

def check_nan():
    for column in ["status", "drinks", "drugs", "diet", "smokes", "sex", "education", "body_type", "religion"]:
        print(column)
        print(df[column].unique())
        print(df[column].isna().any())

#print(check_nan())


new_columns = ["status", "drinks", "drugs", "diet", "smokes", "sex", "education", "religion","body_type" ]

df = df[new_columns]
df.shape

cols = list(df.columns)
for col in cols[:-1]:
    df = pd.get_dummies(df, columns=[col], prefix = [col])
#print(df.head())


features = df.iloc[:,1:len(df.columns)]
target = df.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


'''
scores = []
for i in range(1,201):
  tree = DecisionTreeClassifier(random_state=1, max_depth = i)

  tree.fit(x_train, y_train)

  scores.append(tree.score(x_test,y_test))

plt.plot(range(1,201), scores)

plt.show()
#Best max_depth was 6
#Orginally I didn't put anything for max_depth, but scores too high to the point of overfitting, so I used this method to check the real depth
'''

tree_model = DecisionTreeClassifier(random_state=34, max_depth = 6)
tree_model.fit(x_train, y_train)
print("DT train score:", tree_model.score(x_train, y_train))
print("DT test Score :", tree_model.score(x_test, y_test))
tree_predictions = tree_model.predict(x_train)
print(classification_report(y_train, tree_predictions))


forest_model = RandomForestClassifier(random_state = 34)
forest_model.fit(x_train, y_train)
forest_prediction = forest_model.predict(x_train)
print("RF training score :", forest_model.score(x_train, y_train))
print("RF test score :", forest_model.score(x_test, y_test))
print(classification_report(y_train, forest_prediction))




#Normalizing data for these classifiers!

x = features.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

features_data = pd.DataFrame(x_scaled, columns=features.columns)
target_data = df.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(features_data, target_data, test_size = 0.2, random_state = 42)
Y_train = Y_train.to_numpy().ravel()
Y_test = Y_test.to_numpy().ravel()



'''
accuracies = []
for k in range(1,101):
  classifer = KNeighborsClassifier(n_neighbors = k)
  classifer.fit(X_train, Y_train)
  k_list = range(1,101)
  accuracies.append(classifer.score(X_test, Y_test))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("body_type Accuracy")
plt.show()

#results showed 20 was the best option
'''

classifier = KNeighborsClassifier(n_neighbors = 20)
classifier.fit(X_train, Y_train)
print("KNN train score", classifier.score(X_train, Y_train))
print("KNN test score", classifier.score(X_test, Y_test))
KNC_predict = classifier.predict(X_train)
print(classification_report(Y_train, KNC_predict))



model = LogisticRegression(multi_class = "multinomial", max_iter = 500)
model.fit(X_train, Y_train)
LR_predict = model.predict(X_train)
print("Predict prob:", model.predict_proba(X_train))
print("LR train score:", model.score(X_train,Y_train))
print("LR test score:", model.score(X_test,Y_test))
print(classification_report(Y_train, LR_predict))

'''
#Create a dictionary of possible parameters
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', "linear"]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,Y_train)
print(grid.best_estimator_)
#Results indicate that SVC(C=100, gamma=0.01) was the best option
'''


#vector naive bayes
SVC_model = SVC(kernel = "rbf", gamma = 0.01, C = 100)
SVC_model.fit(X_train, Y_train)
SVC_predict = SVC_model.predict(X_train)
print("SVC train score:", SVC_model.score(X_train, Y_train))
print("SVC test score:", SVC_model.score(X_test, Y_test))
print(classification_report(Y_train, SVC_predict))
