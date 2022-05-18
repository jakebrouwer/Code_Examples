from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import DataFrame,read_csv

class DataLoader:
    """
    Built as a helper class in order to standardize the dataloading process used in other classes
    """

    @staticmethod
    def _load_data_from_json():
        """
        Loads data that has been specified in datasets.json

        Will import data from a given url and specify the target column

        Returns:
            data_sets (dict): A dictionary containing the needed information to load the data via pandas
        """
        from json import load
        from os.path import dirname,join
        from platform import system
        current_path = dirname(__file__) #this line will error in live interpreter such as ipython
        print(current_path)
        sys = system()
        if "win" in sys.lower():
            path = join(current_path,"..\\datasets.json")
        else:
            path = join(current_path,"../datasets.json")

        with open(path,"rt") as file:
            json_data = load(file)

        data_sets = {}
        for data_name,data_info in json_data.items():
            data_sets[data_name] = data_info['url']
        return data_sets

    def __init__(self):
        """
        The constructor for DataLoader
        """
        self._data=None
        self._X_test = None
        self._X_train = None
        self._y_test = None
        self._y_train = None

    def _load_data(self,data= 'breast_cancer',train_ratio:float = 0.5):
        """
        A helper function to actually load the appropriate data

        Args:
            data (optional): the name of the dataset to load. Defaults to 'breast_cancer'.
                    data can also be given as a DataFrame. 
                    In this case it is assumed that the target, or labels column is the last column of the DataFrame.
            train_ratio (float, optional): The ratio of data that will be represented in the training split. Should be less than or equal to 1 . Defaults to 0.5.
        """
        data = data.lower()
        #check if data is a string and if it is a sklearn dataset
        try:
            _data = getattr(datasets,'load_'+data)
            self._data = _data(as_frame=True,return_X_y=True)
            self._X_train,self._X_test,self._y_train,self._y_test = train_test_split(self._data[0],self._data[1],train_size=train_ratio)
        except:
            pass
        #check if data is a DataFrame
        try:
            target_col = data.columns[-1]
            y = data.pop(target_col)
            self._X_train,self._X_test,self.y_train,self._y_test = train_test_split(data,y,train_size=train_ratio)
        except:
            pass
        #check if data is specified in datasets.json
        try:
            data_sets = self._load_data_from_json()
            for data_name in data_sets.keys():
                if data in data_name:
                    self._data = read_csv(data_sets[data_name]["url"])
                    target_col = data_sets[data_name]['target_col']
            y = self._data.pop(target_col)
            self._X_train,self._X_test,self.y_train,self._y_test = train_test_split(self._data,y,train_size=train_ratio)
        #lastly raise TypeError specifying that the data could not be loaded
        except:
            data_sets = self._load_data_from_json()
            names = data_sets.keys()
            raise TypeError('{} could not be loaded. \n\nAvailable datasets are:\
                            \n{} from datasets.json OR\
                            \ndatasets from sklearn.datasets that have the return_X_y parameter'.format(data,names))
