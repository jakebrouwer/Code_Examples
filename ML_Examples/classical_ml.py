from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from .data_loader import DataLoader

class ClassicalML(DataLoader):
    """
    A class housing a suite of classical machine learning functions aimed at quickly finding the best model for a given problem.

    This class inherits from DataLoader in order to load the appropriate data specified in the constructor
    """
    
    @staticmethod
    def _add_models_from_json():
        """
        A method that reads in models from a json file and allows them to be added to the appropriate class variable

        This allows the user to modify the relevant json file to add models they would like to test 
        instead of having to modify code directly

        Returns:
            models (dict): a dictionary of models that will later be used to add the models to the appropriate class variable
        """
        from json import load
        import importlib
        from os.path import dirname,join
        from platform import system
        current_path = dirname(__file__) #this line will error in live interpreter such as ipython
        sys = system() #get operating system
        #modify paths accordingly
        if "win" in sys.lower():
            path = join(current_path,"..\\models.json")
        else:
            path = join(current_path,"../models.json")\

        #read the json file
        with open(path,"rt") as file:
            json_data = load(file)

        #build model dictionary according to the contents of the json file
        models = {}
        for model_name,model_info in json_data.items():
            import_path = model_info['import_path'] 
            default_args = model_info['default_args']
            module = importlib.import_module(import_path) #use importlib to import the model appropriately
            model_class = getattr(module,model_name)
            models[model_name] = model_class(**default_args)
        return models
            
    def __init__(self,data:str='breast_cancer'):
        """
        The constructor for the ClassicalML class

        Args:
            data (str, optional): The name of the dataset to load. Defaults to 'breast_cancer'.
        """
        #initialize class variables
        self.scores = {}
        self.best_pre_optimization_model = None
        self.best_model = None
        self.best_score = None
        # a set of predefined models, models from models.json will be added here
        self.models = {
            'BernoulliNB':BernoulliNB(),
            'GaussianNB':GaussianNB(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'DummyClassifier':DummyClassifier()
        }

        new_models = self._add_models_from_json() #get and build models from models.json

        self.models = {**self.models, **new_models} #build a new model dictionary

        #_load_data() is inherited from DataLoader
        self._load_data(data)

    def find_best_model(self):
        """
        Test all of the models in self.models to find the most performant model out of the box
        """
        #iterate through the models
        for name,model in self.models.items():
            #fit each model with the appropriate data
            model.fit(self._X_train,self._y_train)
            #score the model and record the performance in a dictionary (self.scores)
            self.scores[name] = model.score(self._X_test,self._y_test)
        #get the most performant model out of the box
        self.best_pre_optimization_model = sorted(self.scores.items(), key= lambda x: x[1],reverse=True)[0]

    def optimize_best_model(self,generations:int = 5,population_size=100,cv=None,scoring='accuracy'):
        """
        Once a best general model has been selected this function will optimize that model

        The optimization is done via tpot which implements a genetic algorithm find a pipeline of parameters best suits the provided model

        Args:
            generations (int, optional): Number of iterations to the run pipeline optimization process. Defaults to 5.
            population_size (int, optional): Number of individuals to retain in the genetic programming population every generation.
                    Generally, TPOT will work better when you give it more individuals with which to optimize the pipeline. Defaults to 100.
            cv (optional): Cross-validation strategy used when evaluating pipelines.
                    Possible inputs:
                    integer, to specify the number of folds in a StratifiedKFold,
                    An object to be used as a cross-validation generator, or
                    An iterable yielding train/test splits.. Defaults to None.
            scoring (str, optional): Function used to evaluate the quality of a given pipeline for the classification problem. Defaults to 'accuracy'.
                    The following built-in scoring functions can be used:
                    'accuracy', 'adjusted_rand_score', 'average_precision', 
                    'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 
                    'f1_weighted', 'neg_log_loss', 'precision'
        """
        from tpot import TPOTClassifier
        from sklearn.model_selection import RepeatedStratifiedKFold
        #get the best model
        model = self.models[self.best_pre_optimization_model[0]]
        #set up a tpot configuratoin specifying model as the model to test
        tpot_config = {str(model.__class__).split('\'')[1]:{}}
        #set up crossvalidation
        if not cv:
            cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3)
        #run tpot
        t_pot = TPOTClassifier(verbosity=2,generations=generations,population_size=population_size,cv=cv,scoring=scoring,config_dict=tpot_config)
        t_pot.fit(self._X_train,self._y_train)
        #get the best tpot pipeline
        self.best_model = t_pot.fitted_pipeline_
    
    def score_best_model(self):
        """
        Test the best model to see how it performs
        """
        self.best_model.fit(self._X_train,self._y_train)
        self.best_score = self.best_model.score(self._X_test,self._y_test)
        return(self.best_score)

