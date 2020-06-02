import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 



#Importing the dataset
file = pd.read_csv("world.csv")
file = file.drop( "Time", axis=1)
file.head()



file = file.iloc[0:264,:]
#file.info()



## Replacing '..' by NaN

file = file.replace("..", np.NaN)

file.tail(10)



##Changing data type of all columns from string to float 

file[list(file.columns[2:])] = file[list(file.columns[2:])].astype("float")


#file.info()


file = file.fillna(file.mean())
file.isnull().sum()


from sklearn.cluster import KMeans


x = file.iloc[:,2:]
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)





plt.plot(range(1,11),wcss,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.xticks(np.arange(1,11,step=1))
plt.title("The Elbow Method")
plt.savefig('task2bgraph1.png')
plt.show()


### optimal number of clusters = 3




kmeans = KMeans(n_clusters = 3, random_state=0)
cluster = kmeans.fit_predict(x)
print("clustering: ", cluster)



file["cluster"] = cluster



file["cluster"].value_counts()


### Feature engineering using interaction term pairs


from sklearn.preprocessing import PolynomialFeatures


x = file.iloc[:,2:22]

polynomial = PolynomialFeatures(interaction_only=True)
x = polynomial.fit_transform(x)



#print(x.shape)



x = np.delete(x, 0, axis=1)
print(x.shape)



# Split dataset into training and test set 

y = file.iloc[:,22].values

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=100)

#test_y



#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

print(train_x.shape)


#### Selecting first four interaction terms



train_x = train_x[0:, 21:25]
test_x = test_x[0:, 21:25]


#### Performing 5-NN classification 



from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, y_pred)

print("confusion matrix for 5-NeighborsClassifier: ", cm)



true_value_interaction = y_pred == test_y
true_value_interaction =true_value_interaction.sum()
#true_value_interaction



# Split dataset into training and test set 

x = file.iloc[:,2:22].values

y = file.iloc[:,22].values

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=100)



#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


### performing PCA to select top 4 features



from sklearn.decomposition import PCA

pca = PCA(n_components= 4)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)
explained_variance = pca.explained_variance_ratio_


print("Explained variance: ", explained_variance)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)

cm = confusion_matrix(test_y, y_pred)

print("confusion matrix after PCA: ", cm)




true_value_pca = y_pred == test_y
true_value_pca =true_value_pca.sum()

print("true_value_pca: ", true_value_pca)


# ## Selecting first four features from original dataset


# Split dataset into training and test set 

x = file.iloc[:,2:22].values

y = file.iloc[:,22].values

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=100)



#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)



train_x = train_x[0:, 0:4]
test_x = test_x[0:, 0:4]





classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)

cm = confusion_matrix(test_y, y_pred)
print("confusion matrix 4 features selection: ", cm)





true_value_first4 = y_pred == test_y
true_value_first4 =true_value_first4.sum()
print("true_value_first4: ", true_value_first4)


### Accuracy of classifiers:

print("Accuracy of feature engineering:", round(true_value_interaction/len(test_y),3))
print("Accuracy of PCA:", round(true_value_pca/len(test_y),3))
print("Accuracy of first four features:", round(true_value_first4/len(test_y),3))



accuracy1 = round((true_value_interaction/len(test_y)*100),3)
accuracy2 = round((true_value_pca/len(test_y)*100),3)
accuracy3 = round((true_value_first4/len(test_y)*100),3)



plt.figure(figsize=[8,6])
plt.bar(x=["Feature Engineering","PCA","first four features"],height=[accuracy1,accuracy2,accuracy3], width=0.5,)
plt.title("Accuracy of 5-NN classification")
plt.xlabel("Methods")
plt.ylabel("Accuracy in %")
plt.yticks(np.arange(0,100,5))
plt.savefig('task2bgraph2.png')
plt.show()







