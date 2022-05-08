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
        self.best_model = None
        self.best_parameters = None
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
        return self.best_model

    def optimize_best_model(self,n_trials:int = 100,categorical_suggestions:dict=None,float_suggestions:dict=None,int_suggestions:dict=None):
        import optuna
        from sklearn.metrics import accuracy_score
        if not self.best_model:
            raise AttributeError('A model has not been generated.\nRun find_best_model() to generate a model.\n\n')
        model = self.models[self.best_model[0]]

        def objective(trial):
            opt_params={}

            if categorical_suggestions:
                for key in categorical_suggestions.keys():
                    opt_params[key] = trial.suggest_categorical(key,categorical_suggestions[key])
            if float_suggestions:
                for key in float_suggestions.keys():
                    opt_params[key] = trial.suggest_float(key,**float_suggestions[key])
            if int_suggestions:
                for key in int_suggestions.keys():
                    opt_params[key] = trial.suggest_int(key,**int_suggestions[key])

            if not opt_params == {}:
                model.set_params(**opt_params)
            pred = model.predict(self._X_test)
            return accuracy_score(self._y_test,pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=n_trials)
        trial = study.best_trial
        self.best_parameters = trial.params

