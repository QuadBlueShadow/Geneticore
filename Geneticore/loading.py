import pickle

def load_model(name="model.pickle"):
    with open(name,"rb") as f:
        model = pickle.load(f)
        f.close()
    return model