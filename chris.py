import numpy as np
import chris_krimskrams

def ldData(data_dict, **args):
    with open("./original_data/X_train.csv") as file:
        list = file.readlines()[1:]
    list = [np.array(x.split(","), dtype=np.float32) for x in list]
    X_train = np.full((len(list),len(max(list,key = lambda x: len(x)))), np.nan, dtype=np.float32)
    for i,j in enumerate(list):
        X_train[i][0:len(j)] = j
    data_dict["X_train"] = X_train[:,1:]

    data_dict["y_train"] = np.loadtxt("./original_data/y_train.csv",
                               dtype=np.intc,
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

def mlpClassification(data_dict, mlpClassification_epochs, mlpClassification_useValidationSet, mlpClassification_makePrediction, **args):
  #data doesnt have to be normalized.

  assert "X_train" in data_dict.keys()
  assert "y_train" in data_dict.keys()
  if mlpClassification_useValidationSet:
    assert "X_val" in data_dict.keys()
    assert "y_val" in data_dict.keys()
  if mlpClassification_makePrediction:
    assert "X_test" in data_dict.keys()


  if mlpClassification_useValidationSet:
    data_dict["train_losses"], data_dict["val_losses"], predict_funct = chris_krimskrams.train(mlpClassification_epochs, data_dict["X_train"], data_dict["y_train"], data_dict["X_val"], data_dict["y_val"], 32)
  else:
    data_dict["train_losses"], predict_funct = chris_krimskrams.train(mlpClassification_epochs, data_dict["X_train"], data_dict["y_train"], None, None, 32)

  if mlpClassification_makePrediction:
    data_dict["y_test"] = predict_funct(data_dict["X_test"])


  return data_dict
