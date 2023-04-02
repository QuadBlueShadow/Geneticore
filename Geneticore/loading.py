import pickle

def load_model():
    pickle_in = open("model.pickle","rb")
    model = pickle.load(pickle_in)
    return model