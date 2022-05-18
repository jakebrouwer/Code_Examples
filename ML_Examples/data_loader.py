from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import DataFrame,read_csv

class DataLoader:

    @staticmethod
    def _load_data_from_json():
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

    def __init__(self,data:str = 'breast_cancer'):
        data = data.lower()
        if isinstance(data,str):
            _data = getattr(datasets,'load_'+data)
            self._data = _data(as_frame=True,return_X_y=True)
            self._X_train,self._X_test,self._y_train,self._y_test = train_test_split(self._data[0],self._data[1])
        elif isinstance(data,DataFrame):
            target_col = data.columns[-1]
            y = data.pop(target_col)
            self._X_train,self._X_test,self.y_train,self._y_test = train_test_split(data,y)
        elif isinstance(data,str):
            data_sets = self._load_data_from_json()
            for data_name in data_sets.keys():
                if data in data_name:
                    self._data = read_csv(data_sets[data_name])
        else:
            data_sets = self._load_data_from_json()
            names = data_sets.keys()
            raise TypeError('{} could not be loaded. \n\nAvailable datasets are:\
                            \n{} from datasets.json OR\
                            \ndatasets from sklearn.datasets that have the return_X_y parameter'.format(data,names))
