import numpy as np

def ldData(data_dict, **args):
    with open("./original_data/X_train.csv") as file:
        list = file.readlines()[1:]
    list = [np.array(x.split(","), dtype=np.float32) for x in list]
    X_train = np.full((len(list),len(max(list,key = lambda x: len(x)))), np.nan, dtype=np.float32)
    for i,j in enumerate(list):
        X_train[i][0:len(j)] = j
    data_dict["X_train"] = X_train[:,1:]

    data_dict["y_train"] = np.loadtxt("./original_data/y_train.csv",
                               dtype=np.uint,
                               delimiter=',',
                               skiprows=1)[:, 1:]

    with open("./original_data/X_test.csv") as file:
        list = file.readlines()[1:]
    list = [np.array(x.split(","), dtype=np.float32) for x in list]
    X_test = np.full((len(list),len(max(list,key = lambda x: len(x)))), np.nan, dtype=np.float32)
    for i,j in enumerate(list):
        X_test[i][0:len(j)] = j
    data_dict["X_test"] = X_test[:,1:]

    return data_dict