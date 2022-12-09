## Important libraries 

from types import GeneratorType
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import Decis



## Read in data for my computer 
# do not use the github link, save the data in a different file 

names = pd.read_csv("names_year.txt")


## Getting rid of commas in a graph 

names = pd.read_csv("names_year.txt", parse_dates=['year'])

 ## Making a vertical line + a chart 

 brooke= (names
.query('name == "Brooke"'))
chart = (alt.Chart(brooke).encode(
    x =alt.X ("year", title = "year"), 
    y= alt.Y ("Total")
).mark_line()

) 


bday_data = brooke.query('year == 2001')
bday_data

bday_chart = (alt.Chart(bday_data).encode(
    x = alt.X("year", title = "Year"), 
    y = alt.Y("Total")
).mark_point(color = "red")
)

chart + bday_chart

## How to make the graph between two dates 

names_christian = (names.query('name == ["Mary", "Martha", "Peter", "Paul"]').query("year >= 1920 & year <= 2000"))



christian = (alt.Chart(names_christian).mark_line().encode(
    x = alt.X("year"),
    y = alt.Y("Total"), 
    color = alt.Color("name")
    )
)

christian 

## Read in SQL data 
con = sqlite3.connect('lahmansbaseballdb.sqlite')

## Join two datasets together in SQL
df = pd.read_sql_query("""
SELECT DISTINCT Salaries.PlayerID, CollegePlaying.SchoolID, Salaries.teamID, Salaries.Salary, Salaries.yearID
FROM Salaries
JOIN CollegePlaying ON Salaries.playerID = CollegePlaying.playerID
WHERE CollegePlaying.schoolID LIKE '%BYUI%' 
ORDER BY salary DESC
""",con)

df

## Clean up Data 
clean2 = (flights
    .replace(-999, np.nan)
    .replace("1500+", "1500")
    .query("month != 'n/a'")
    .assign(

#.replace allows us to replace a value with a different value 

## Machine Learning
#When doing machine learning, it is important to 
#explore the data by making graphs first 

## Machine learning code 
## Can change the filter and the decision tree classifier 
# %%
# Load data
dwellings_ml = pd.read_csv("dwellings_ml.txt")

#%%
# Separate the features (X) and targets (Y)
x = dwellings_ml.filter(["basement","stories","numbaths", "nocars", "sprice", "arcstyle", 'livearea', 'tasp'])
y = dwellings_ml[["before1980"]]

#%% Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .021, random_state = 190)

# Create a decision tree
classifier_DT = model = GradientBoostingClassifier(max_depth = 13)

# Fit the decision tree
classifier_DT.fit(x_train, y_train)

# Test the decision tree (make predictions)
y_predicted_DT = classifier_DT.predict(x_test)

# Evaluate the decision tree
print("Accuracy:", metrics.accuracy_score(y_test, y_predicted_DT)

## Confusion Matrix 

#%%
# a confusion matrix
print(metrics.confusion_matrix(y_test, y_predicted_DT))

#%%
# this one might be easier to read
print(pd.crosstab(y_test.before1980, y_predicted_DT, rownames=['True'], colnames=['Predicted'], margins=True))

#%%
# visualize a confusion matrix
# requires '.' to be installed
metrics.plot_confusion_matrix(classifier_DT, x_test, y_test)

## Feature Importance 
classifier_DT.feature_importances_

#%%
feature_df = pd.DataFrame({'features':x.columns, 'importance':classifier_DT.feature_importances_})
feature_df