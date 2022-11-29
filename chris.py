import numpy as np
import chris_krimskrams
from biosppy.signals.ecg import christov_segmenter, extract_heartbeats
from biosppy.signals.tools import normalize, synchronize
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def ldData(data_dict, **args):
    with open("./original_data/X_train.csv") as file:
        list = file.readlines()[1:]
    #list = [np.array(x.split(","), dtype=np.float32) for x in list]
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
    #list = [np.array(x.split(","), dtype=np.float32) for x in list]
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


def biosppyRPeaks(data_dict, **args):
  assert "X_train" in data_dict.keys()
  assert "X_test" in data_dict.keys()
  
  train_peaks = []

  for ts in tqdm(data_dict["X_train"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    train_peaks.append(christov_segmenter(ts[:firstnan],sampling_rate=300)["rpeaks"])

  test_peaks = []

  for ts in tqdm(data_dict["X_test"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    test_peaks.append(christov_segmenter(ts[:firstnan],sampling_rate=300)["rpeaks"])

  data_dict["train_rpeaks"] = train_peaks
  data_dict["test_rpeaks"] = train_peaks

  return data_dict


def biosppyExtract(data_dict, **args):
  assert "X_train" in data_dict.keys()
  assert "X_test" in data_dict.keys()
  assert "train_rpeaks" in data_dict.keys()
  assert "test_rpeaks" in data_dict.keys()
  
  heartbeat_templates_train = []

  for i, ts in enumerate(data_dict["X_train"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    heartbeat_templates_train.append(extract_heartbeats(ts[:firstnan],data_dict["train_rpeaks"][i],sampling_rate=300)[0])

  heartbeat_templates_test = []

  for i, ts in enumerate(data_dict["X_test"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    heartbeat_templates_test.append(extract_heartbeats(ts[:firstnan],data_dict["test_rpeaks"][i],sampling_rate=300)[0])

  data_dict["heartbeat_templates_train"] = heartbeat_templates_train
  data_dict["heartbeat_templates_test"] = heartbeat_templates_test
  
  return data_dict


def plotNormHeartbeat(data_dict,plotNormHeartbeat_index=None,**args):

    assert "heartbeat_templates_train" in data_dict.keys()

    if plotNormHeartbeat_index==None:
      plotNormHeartbeat_index = randint(0,len(data_dict["heartbeat_templates_train"])-1)

    mean = np.mean(data_dict["heartbeat_templates_train"][plotNormHeartbeat_index], axis=0)
    std = np.std(data_dict["heartbeat_templates_train"][plotNormHeartbeat_index], axis=0)

    plt.plot(mean, color="green")
    plt.fill_between(np.arange(0,len(mean)),mean-std,mean+std, color="green", alpha=0.1)

    plt.show()

    return data_dict


def normalizeTemplates(data_dict,normalizeTemplates_normOverEntireTimeseries=False,**args):

    assert "heartbeat_templates_train" in data_dict.keys()
    assert "heartbeat_templates_test" in data_dict.keys()

    heartbeat_templates_train= []
    heartbeat_templates_test= []
    if normalizeTemplates_normOverEntireTimeseries:
      for l in data_dict["heartbeat_templates_train"]:
        heartbeat_templates_train.append(normalize(l)["signal"])
      for l in data_dict["heartbeat_templates_test"]:
        heartbeat_templates_test.append(normalize(l)["signal"])
    else:
      for l in data_dict["heartbeat_templates_test"]:
        heartbeat_templates_train.append([normalize(x)["signal"] for x in l])
      for l in data_dict["heartbeat_templates_test"]:
        heartbeat_templates_test.append([normalize(x)["signal"] for x in l])

    data_dict["heartbeat_templates_train"] = heartbeat_templates_train
    data_dict["heartbeat_templates_test"] = heartbeat_templates_test
    

    return data_dict


def medmeanFeatures(data_dict, medmeanFeatures_useMedian=True,medmeanFeatures_useMedianSTD=True,**args):
  assert "heartbeat_templates_train" in data_dict.keys()
  assert "heartbeat_templates_test" in data_dict.keys()

  mitte_features_train = []
  mitte_features_test = []
  for templates in data_dict["heartbeat_templates_train"]:
    np_template = np.array(templates, dtype=np.float32)
    if medmeanFeatures_useMedian:
      mitte_features_train.append(np.median(np_template,axis=0))
    else:
      mitte_features_train.append(np.mean(np_template,axis=0))
  for templates in data_dict["heartbeat_templates_test"]:
    np_template = np.array(templates, dtype=np.float32)
    if medmeanFeatures_useMedian:
      mitte_features_test.append(np.median(np_template,axis=0))
    else:
      mitte_features_test.append(np.mean(np_template,axis=0))
  
  abweichung_features_train=[]
  abweichung_features_test=[]
  for i, templates in enumerate(data_dict["heartbeat_templates_train"]):
    np_template = np.array(templates, dtype=np.float32)-mitte_features_train[i]
    if medmeanFeatures_useMedianSTD:
      abweichung_features_train.append(np.median(np.abs(np_template),axis=0))
    else:
      abweichung_features_train.append(np.sqrt(np.mean(np.square(np_template),axis=0)))
  for i, templates in enumerate(data_dict["heartbeat_templates_test"]):
    np_template = np.array(templates, dtype=np.float32)-mitte_features_test[i]
    if medmeanFeatures_useMedianSTD:
      abweichung_features_test.append(np.median(np.abs(np_template),axis=0))
    else:
      abweichung_features_test.append(np.sqrt(np.mean(np.square(np_template),axis=0)))

  mitte_features_train = np.array(mitte_features_train)
  mitte_features_test = np.array(mitte_features_test)

  abweichung_features_train=np.array(abweichung_features_train)
  abweichung_features_test=np.array(abweichung_features_test)


  data_dict["X_train"] = np.append(mitte_features_train,abweichung_features_train,axis=1)
  data_dict["X_test"] = np.append(mitte_features_test,abweichung_features_test,axis=1)

  return data_dict


def makeTrainValSet(data_dict,makeTrainValSet_valPercent=0.1, **args):
  assert "X_train" in data_dict.keys()
  assert "X_test" in data_dict.keys()
  assert "y_train" in data_dict.keys()

  data_dict["X_train"], data_dict["X_val"], data_dict["y_train"], data_dict["y_val"] = train_test_split(data_dict["X_train"],  data_dict["y_train"], test_size=makeTrainValSet_valPercent, random_state=42, shuffle=True)
  
  assert len(data_dict["y_train"].shape) == 2
  assert len(data_dict["y_val"].shape) == 2

  assert data_dict["y_train"].shape[1] == 1
  assert data_dict["y_val"].shape[1] == 1

  return data_dict

'''def synchronizeTemplates(data_dict,**args):

    assert "heartbeat_templates_train" in data_dict.keys()
    assert "heartbeat_templates_test" in data_dict.keys()

    heartbeat_templates_train= []
    heartbeat_templates_test= []
    for l in data_dict["heartbeat_templates_test"]:
      heartbeat_templates_train.append([synchronize(l[0],x)["synch_y"] for x in l])
    for l in data_dict["heartbeat_templates_test"]:
      heartbeat_templates_test.append([synchronize(l[0],x)["synch_y"] for x in l])

    data_dict["heartbeat_templates_train"] = heartbeat_templates_train
    data_dict["heartbeat_templates_test"] = heartbeat_templates_test
    

    return data_dict
'''