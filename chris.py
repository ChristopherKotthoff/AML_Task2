import numpy as np
import chris_krimskrams
from biosppy.signals.ecg import christov_segmenter, extract_heartbeats
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt

def ldData(data_dict, **args):
    with open("./original_data/X_train.csv") as file:
        list = file.readlines()[1:]
    #list = [np.array(x.split(","), dtype=np.float32) for x in list]
    list = [np.array(x.split(","), dtype=np.intc) for x in list]
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
    #list = [np.array(x.split(","), dtype=np.float32) for x in list]
    list = [np.array(x.split(","), dtype=np.intc) for x in list]
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


def biosppyRPeaks(data_dict, **args):
  assert "X_train" in data_dict.keys()
  assert "X_test" in data_dict.keys()
  
  train_peaks = []

  for ts in tqdm(data_dict["X_train"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    train_peaks.append(christov_segmenter(ts[:firstnan],sampling_rate=300))

  test_peaks = []

  for ts in tqdm(data_dict["X_test"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    test_peaks.append(christov_segmenter(ts[:firstnan],sampling_rate=300))

  data_dict["train_rpeaks"] = train_peaks
  data_dict["test_rpeaks"] = train_peaks

  return data_dict


def biosppyExtract(data_dict, **args):
  assert "X_train" in data_dict.keys()
  assert "X_test" in data_dict.keys()
  assert "train_rpeaks" in data_dict.keys()
  assert "test_rpeaks" in data_dict.keys()
  
  heartbeat_templates_train = []

  for ts in tqdm(data_dict["X_train"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    heartbeat_templates_train.append(extract_heartbeats(ts[:firstnan],data_dict["train_rpeaks"],sampling_rate=300)[0])

  heartbeat_templates_test = []

  for ts in tqdm(data_dict["X_test"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    heartbeat_templates_test.append(extract_heartbeats(ts[:firstnan],data_dict["test_rpeaks"],sampling_rate=300)[0])

  data_dict["heartbeat_templates_train"] = heartbeat_templates_train
  data_dict["heartbeat_templates_test"] = heartbeat_templates_test
  
  return data_dict


def plotNormHeartbeat(data_dict,index=None,**args):

    assert "heartbeat_templates_train" in data_dict.keys()

    if index==None:
      index = randint(0,len(data_dict["heartbeat_templates_train"])-1)

    mean = np.mean(data_dict["heartbeat_templates_train"][index], axis=0)
    std = np.std(data_dict["heartbeat_templates_train"][index], axis=0)

    plt.plot(mean, color="green")
    plt.fill_between(np.arange(0,len(mean)),mean-std,mean+std, color="green", alpha=0.1)

    return data_dict
