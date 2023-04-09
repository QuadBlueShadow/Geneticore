import pickle

def load_model(name="model.pickle"):
    pickle_in = open(name,"rb")
    model = pickle.load(pickle_in)
    return model