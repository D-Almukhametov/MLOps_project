from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle


class ClassModel:
    def __init__(self):
        self.trained_models = {}
    

    def get_available_classes(self):
        return ['RandomForest', 'LogisticRegression']
    

    def train(self, model_name, parameters, X, y):
        if model_name == 'RandomForest':
            model = RandomForestClassifier(**parameters)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(**parameters)
        
        model.fit(X, y)
        self.trained_models[model_name] = model
        PATH = 'models/' + model_name + '.pickle'
        with open(PATH, mode='wb') as f:
            pickle.dump(model, f)
        return f'{model_name} была обучена'
    
    def predict(self, model_name, X):
        model = self.trained_models[model_name]
        return model.predict(X).tolist()