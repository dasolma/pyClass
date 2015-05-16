__author__ = 'dasolma'


# coding: utf-8

# In[17]:

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.tools.plotting import scatter_matrix, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# In[83]:

def representacion_grafica(datos,caracteristicas, objetivo, clases, c1, c2):

    for tipo,marca,color in zip(range(len(clases)),"soD","rgb"):

        plt.scatter(datos[objetivo == tipo,c1],
                    datos[objetivo == tipo,c2],
                    marker=marca,c=color)

    plt.ylabel(caracteristicas[c2])
    plt.xlabel(caracteristicas[c1])
    plt.legend(clases,loc='center left', bbox_to_anchor=(1, 0.5))




# In[84]:

iris = load_iris()

X_iris, y_iris = iris.data, iris.target
X_names, y_names = iris.feature_names, iris.target_names
X_names


# In[85]:

X_iris =  np.delete(X_iris, np.s_[X_names.index('sepal length (cm)'),X_names.index('sepal width (cm)')], axis=1)
X_names = ['petal length (cm)','petal width (cm)']
X_names


# In[86]:

representacion_grafica(X_iris, iris.feature_names, y_iris, iris.target_names,
                       X_names.index('petal length (cm)'),
                       X_names.index('petal width (cm)'))
plt.show()

# In[87]:

X_train, X_test, y_train, y_test = train_test_split(X_iris,y_iris,test_size = 0.25)
representacion_grafica(X_test, iris.feature_names, y_test, iris.target_names,
                       X_names.index('petal length (cm)'),
                       X_names.index('petal width (cm)'))
plt.show()

# In[88]:

representacion_grafica(X_train, iris.feature_names, y_train, iris.target_names,
                       X_names.index('petal length (cm)'),
                       X_names.index('petal width (cm)'))

plt.show()
# In[89]:

normalizador = StandardScaler().fit(X_train)
Xn_train = normalizador.transform(X_train)
Xn_test = normalizador.transform(X_test)
representacion_grafica(Xn_train, iris.feature_names, y_train, iris.target_names,
                       X_names.index('petal length (cm)'),
                       X_names.index('petal width (cm)'))
plt.show()

# In[90]:

clasificador = SGDClassifier().fit(Xn_train,y_train)
clasificador.coef_


# In[91]:

clasificador.intercept_


# In[103]:

X = Xn_train  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = y_train
colors = "bry"

# shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = Xn_train[:, 0].min() - 1, Xn_train[:, 0].max() + 1
y_min, y_max = Xn_train[:, 1].min() - 1, Xn_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clasificador.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

# Plot also the training points
for i, color in zip(clasificador.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.Paired)
plt.title("Decision surface of multi-class SGD")
plt.axis('tight')

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clasificador.coef_
intercept = clasificador.intercept_

colors = "bry"
def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)],
             ls="--", color=color)

for i, color in zip(clasificador.classes_, colors):
    plot_hyperplane(i, color)
plt.show()

# In[94]:

y_test_pred = clasificador.predict(normalizador.transform(X_test))


# In[95]:

metrics.accuracy_score(y_test,y_test_pred)


# In[105]:

print metrics.classification_report(y_test,y_test_pred)


# In[106]:

print metrics.confusion_matrix(y_test,y_test_pred)


# In[110]:

modelo = Pipeline([ ('normalizador', StandardScaler()), ('modelolineal', SGDClassifier())])
modelo.fit(X_train,y_train)
y_test_pred = modelo.predict(X_test)
print metrics.confusion_matrix(y_test,y_test_pred)


# In[113]:

#CROSS VALIDATION
#split the train data
from sklearn.cross_validation import KFold
kfold5 = KFold(X_train.shape[0], 5, shuffle = True)


# In[140]:

from sklearn.cross_validation import cross_val_score
vals = cross_val_score(modelo, X_iris, y_iris, cv=kfold5)
vals


# In[141]:

np.mean(vals)


# In[142]:

import scipy

#standard error
scipy.stats.sem(vals)


# In[152]:

#Obtaining predictions by cross-validation
from sklearn import cross_validation
predicted = cross_validation.cross_val_predict(modelo, X_iris,
                                                y_iris, cv=5)
print metrics.accuracy_score(y_iris, predicted)


# In[ ]:


