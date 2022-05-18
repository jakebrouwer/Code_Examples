from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from .data_loader import DataLoader

class ClassicalML(DataLoader):
    
    @staticmethod
    def _add_models_from_json():
        from json import load
        import importlib
        from os.path import dirname,join
        from platform import system
        current_path = dirname(__file__) #this line will error in live interpreter such as ipython
        print(current_path)
        sys = system()
        if "win" in sys.lower():
            path = join(current_path,"..\\models.json")
        else:
            path = join(current_path,"../models.json")

        with open(path,"rt") as file:
            json_data = load(file)
        models = {}
        for model_name,model_info in json_data.items():
            import_path = model_info['import_path']
            default_args = model_info['default_args']
            module = importlib.import_module(import_path)
            model_class = getattr(module,model_name)
            models[model_name] = model_class(**default_args)
        return models
            

    def __init__(self,data='breast_cancer'):
        self.scores = {}
        self.best_pre_optimization_model = None
        self.best_model = None
        self.best_score = None

        self.models = {
            'BernoulliNB':BernoulliNB(),
            'GaussianNB':GaussianNB(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'DummyClassifier':DummyClassifier()
        }

        new_models = self._add_models_from_json()

        self.models = {**self.models, **new_models}

    def find_best_model(self):
        from sklearn.metrics import accuracy_score,f1_score,precision_score
        for name,model in self.models.items():
            model.fit(self._X_train,self._y_train)
            self.scores[name] = model.score(self._X_test,self._y_test)
        self.best_pre_optimization_model = sorted(self.scores.items(), key= lambda x: x[1],reverse=True)[0]
        return self.best_pre_optimization_model

    def optimize_best_model(self,generations:int = 5,population_size=100,cv=None,scoring='accuracy'):
        from tpot import TPOTClassifier
        from sklearn.model_selection import RepeatedStratifiedKFold
        model = self.models[self.best_pre_optimization_model[0]]
        tpot_config = {str(model.__class__).split('\'')[1]:{}}
        if not cv:
            cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3)
        t_pot = TPOTClassifier(verbosity=2,generations=generations,population_size=population_size,cv=cv,scoring=scoring,config_dict=tpot_config)
        t_pot.fit(self._X_train,self._y_train)
        #t_pot.export(path+'best_model.py')
        self.best_model = t_pot.fitted_pipeline_
    
    def score_best_model(self):
        self.best_model.fit(self._X_train,self._y_train)
        self.best_score = self.best_model.score(self._X_test,self._y_test)
        return(self.best_score)

if __name__ == "__main__":
    ml = ClassicalML()
    ml.find_best_model()
    ml.optimize_best_model()
    ml.score_best_model()
