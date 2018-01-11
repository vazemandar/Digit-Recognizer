import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.grid_search import GridSearchCV


scores1=[]
scores2=[]
scores3=[]
data=pd.read_csv('C:/Users/mandar/Desktop/KaggleData/train.csv')
testData=pd.read_csv('C:/Users/mandar/Desktop/KaggleData/test.csv')
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
testDf=testData.iloc[:,0:]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=4)

# One of the simpleton methods of tuning parametres
# Creating graphs for accuracy and picking out the best parametres
def RFPerformanceGraph():
    k_range = list(range(1,1000))
    for k in k_range:
        clf= RandomForestClassifier(n_estimators=k,max_depth=120,min_samples_leaf=3)
        clf.fit(x_train,y_train)
        predicted = clf.predict(x_test) 
        scores1.append(metrics.accuracy_score(y_test, predicted))
    ymax = max(scores1)
    xpos = scores1.index(ymax)
    xmax = k_range[xpos]
    print('nestimator:', ymax, xmax) 
    
    for k in k_range:
        clf2= RandomForestClassifier(n_estimators=100,max_depth=k,min_samples_leaf=3)
        clf2.fit(x_train,y_train)
        predicted2 = clf2.predict(x_test)
        scores2.append(metrics.accuracy_score(y_test, predicted2))
    ymax = max(scores2)
    xpos = scores2.index(ymax)
    xmax = k_range[xpos]
    print('max_depth', ymax, xmax)
    
    
    for k in k_range:
        clf3= RandomForestClassifier(n_estimators=100,max_depth=120,min_samples_leaf=k)
        clf3.fit(x_train,y_train)
        predicted3 = clf3.predict(x_test)
        scores3.append(metrics.accuracy_score(y_test, predicted3))
    ymax = max(scores3)
    xpos = scores3.index(ymax)
    xmax = k_range[xpos]
    print('min_sample_split', ymax, xmax)
    
    plt.plot(k_range, scores1)
    plt.plot(k_range, scores2)
    plt.plot(k_range, scores3)
    plt.xlabel('Value  for Random Forest')
    plt.ylabel('Testing Values')
    plt.legend(['nestimator', 'max_depth','min_sample_Leaf'], loc='upper center')

RFPerformanceGraph()
#Using  grid search trying to tune the parametres
def RFgridsearch():
    # the list for range is just experimental example, it takes a lot of time
    #we could use multiple of tens or something like that
    parameters = {"max_depth": list(range(2,1000))
            ,"min_samples_split" :list(range(2,1000))
            ,"n_estimators" : list(range(2,1000))
            ,"min_samples_leaf": list(range(2,1000))
            ,"max_features": (4,5,6,"sqrt")
            ,"criterion": ('gini','entropy')}

    model = GridSearchCV(RandomForestClassifier(),parameters, n_jobs = 3, cv = 10)
    model_fit = model.fit(x_train,y_train)  
    tuned_parameters = model_fit.best_params_ 
    return tuned_parameters




bestparam=RFgridsearch()
# sustituting the value for best parametres 
clf5= RandomForestClassifier(n_estimators=bestparam["n_estimators"],max_depth=bestparam["max_depth"],min_samples_leaf=bestparam["min_samples_leaf"])
clf5.fit(x_train,y_train)
predicted5 = clf5.predict(testDf)
res=pd.Series(predicted5)
