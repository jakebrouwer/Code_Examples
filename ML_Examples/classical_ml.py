import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

class ClassicalML:
    
    def __init__(self,data='breast_cancer'):
        self.scores = {}
        self.best_pre_optimization_model = None
        self.best_model = None
        self.best_score = None
        if isinstance(data,str):
            _data = getattr(datasets,'load_'+data)
            self._data = _data(as_frame=True,return_X_y=True)
            self._X_train,self._X_test,self._y_train,self._y_test = train_test_split(self._data[0],self._data[1])
        elif isinstance(data,pd.DataFrame):
            target_col = data.columns[-1]
            y = data.pop(target_col)
            self._X_train,self._X_test,self.y_train,self._y_test = train_test_split(data,y)
        else:
            raise TypeError('{} could not be loaded. \n\nAvailable datasets are datasets from sklearn.datasets that have the return_X_y parameter'.format(data))

        self.models = {
            'LogisticRegression':LogisticRegression(),
            'Perceptron':Perceptron(),
            'KNeighborsClassifier':KNeighborsClassifier(),
            'BernoulliNB':BernoulliNB(),
            'GaussianNB':GaussianNB(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'DummyClassifier':DummyClassifier()
        }

    def add_model(self,model:'sklearn_like',name:'str'):
        from sklearn.base import is_classifier
        if is_classifier(model) and hasattr(model,'fit'):
            self.models[name]=model()

    def find_best_model(self):
        from sklearn.metrics import accuracy_score,f1_score,precision_score
        for name,model in self.models.items():
            model.fit(self._X_train,self._y_train)
            self.scores[name] = model.score(self._X_test,self._y_test)
        self.best_model = sorted(self.scores.items(), key= lambda x: x[1],reverse=True)[0]
        return self.best_pre_optimization_model

    def optimize_best_model(self,generations:int = 5,population_size=100,cv=None,scoring='accuracy'):
        from tpot import TPOTClassifier
        from sklearn.model_selection import RepeatedStratifiedKFold
        if not cv:
            cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3)
        t_pot = TPOTClassifier(verbosity=2,generations=generations,population_size=population_size,cv=cv,scoring=scoring)
        t_pot.fit(self._X_train,self._y_train)
        #t_pot.export(path+'best_model.py')
        self.best_model = t_pot.fitted_pipeline_
    
    def score_best_model(self):
        self.best_model.fit(self._X_train,self._y_train)
        self.best_score = self.best_model.score(self._X_test,self._y_test)
        return(self.best_score)

