import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 





#Importing the dataset
file1 = pd.read_csv("world.csv")
file1 = file1.drop(["Country Name", "Time"], axis=1)
file1.head()



#Importing the dataset
file2 = pd.read_csv("life.csv")
file2.head()





#Merging the two files 
file2 = file2.drop("Year", axis=1)
file = file2.merge(file1, on="Country Code", copy=False)
file.head()





#Inspecting the dataset
#file.info()





## Replacing '..' by NaN

file = file.replace("..", np.NaN)

file.tail(10)





file.columns




##Changing data type of all 20 features from string to float 

file[list(file.columns[3:])] = file[list(file.columns[3:])].astype("float")





file.describe()


# ### - Low represented by 0
# 
# ### - Medium represented by 1
# 
# ### - High represented by 2




# Assigning integer values to Life expectancy column

file["Life expectancy at birth (years)"] = file["Life expectancy at birth (years)"].str.replace("Low","0")
file["Life expectancy at birth (years)"] = file["Life expectancy at birth (years)"].str.replace("Medium","1")
file["Life expectancy at birth (years)"] = file["Life expectancy at birth (years)"].str.replace("High","2")
file["Life expectancy at birth (years)"] = file["Life expectancy at birth (years)"].astype(int)





file["Life expectancy at birth (years)"].unique()





# Split dataset into training and test set 

x = file.iloc[:,3:].values
y = file.iloc[:,2].values

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=1/3,random_state=100)





#Replacing NaN values with median imputation 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer.fit(train_x)
train_x = imputer.transform(train_x)
test_x = imputer.transform(test_x)
imputer.statistics_




#Scaling features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
scaler.mean_





data = {'Feature': file.columns[3:],
        'Median': imputer.statistics_.tolist(),
        'Mean': scaler.mean_.tolist(),
        "Variance": scaler.var_.tolist()}

df = pd.DataFrame(data, columns=["Feature","Median","Mean","Variance"])

df.style.set_properties(subset=["Variance"], **{'width': '150px'})





df["Median"] = round(df["Median"], 3)
df["Mean"] = round(df["Mean"], 3)
df["Variance"] = round(df["Variance"], 3)





df




df.to_csv("task2a.csv", index=False)



# ### Performing K Neighbours classification with k=5




from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)





from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, y_pred)
cm





true_value_k5 = y_pred == test_y
true_value_k5 =true_value_k5.sum()
true_value_k5


#### Performing K Neighbours classification with k=10




classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)

cm = confusion_matrix(test_y, y_pred)
cm





true_value_k10 = y_pred == test_y
true_value_k10 = true_value_k10.sum()
true_value_k10


#### Decision Tree Classifier




from sklearn.tree import DecisionTreeClassifier

classifer = DecisionTreeClassifier(max_depth=4)
classifer.fit(train_x,train_y)
pred = classifer.predict(test_x)

cm = confusion_matrix(test_y, pred)
cm





true_value_dt = pred == test_y
true_value_dt = true_value_dt.sum()
true_value_dt


# ### Accuracy of classifiers:
# 




print("Accuracy of decision tree:", round(true_value_dt/len(test_y),3))
print("Accuracy of k-nn (k=5):", round(true_value_k5/len(test_y),3))
print("Accuracy of k-nn (k=10):", round(true_value_k10/len(test_y),3))
