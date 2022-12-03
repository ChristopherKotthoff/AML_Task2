import numpy as np
import chris_krimskrams
from biosppy.signals.ecg import christov_segmenter, extract_heartbeats, ecg
from biosppy.signals.tools import normalize, synchronize
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score
import math


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
    data_dict["y_val_hat"]=predict_funct(data_dict["X_val"])
    data_dict["y_train_hat"]=predict_funct(data_dict["X_train"])
  else:
    data_dict["train_losses"], predict_funct = chris_krimskrams.train(mlpClassification_epochs, data_dict["X_train"], data_dict["y_train"], None, None, 32)
    data_dict["y_train_hat"]=predict_funct(data_dict["X_train"])

  if mlpClassification_makePrediction:
    data_dict["y_test"] = predict_funct(data_dict["X_test"])

  if mlpClassification_useValidationSet:
    data_dict["y_val_predicted"] = predict_funct(data_dict["X_val"])


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


def NO_DISPLAY_plotNormHeartbeat(data_dict,plotNormHeartbeat_index=None,**args):

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


def medmeanFeatures(data_dict, medmeanFeatures_useMedian=True,medmeanFeatures_useMedianSTD=True, medmeanFeatures_shrinkingFactor=1,**args):
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
  
  normal = len(mitte_features_train[0])
  #print(mitte_features_train)
  #print(normal)
  #for i in mitte_features_train:
  #  if len(i)!=normal:
  #    print(len(i))

  mitte_features_train = np.array(mitte_features_train, dtype=np.float32)
  mitte_features_test = np.array(mitte_features_test, dtype=np.float32)

  abweichung_features_train=np.array(abweichung_features_train, dtype=np.float32)
  abweichung_features_test=np.array(abweichung_features_test, dtype=np.float32)

  if medmeanFeatures_shrinkingFactor != 1:
    mitte_features_train_shrunk = np.empty((mitte_features_train.shape[0],math.ceil(mitte_features_train.shape[1]/float(medmeanFeatures_shrinkingFactor))), dtype=np.float32)
    mitte_features_test_shrunk = np.empty((mitte_features_test.shape[0],math.ceil(mitte_features_test.shape[1]/float(medmeanFeatures_shrinkingFactor))), dtype=np.float32)
    abweichung_features_train_shrunk = np.empty((abweichung_features_train.shape[0],math.ceil(abweichung_features_train.shape[1]/float(medmeanFeatures_shrinkingFactor))), dtype=np.float32)
    abweichung_features_test_shrunk = np.empty((abweichung_features_test.shape[0],math.ceil(abweichung_features_test.shape[1]/float(medmeanFeatures_shrinkingFactor))), dtype=np.float32)

    for col_index, i in enumerate(range(0,mitte_features_train.shape[1],medmeanFeatures_shrinkingFactor)):
      mitte_features_train_shrunk[:,col_index:col_index+1] = np.mean(mitte_features_train[:,i:i+medmeanFeatures_shrinkingFactor],axis=1,keepdims=True)
      mitte_features_test_shrunk[:,col_index:col_index+1] = np.mean(mitte_features_test[:,i:i+medmeanFeatures_shrinkingFactor],axis=1,keepdims=True)
      abweichung_features_train_shrunk[:,col_index:col_index+1] = np.mean(abweichung_features_train[:,i:i+medmeanFeatures_shrinkingFactor],axis=1,keepdims=True)
      abweichung_features_test_shrunk[:,col_index:col_index+1] = np.mean(abweichung_features_test[:,i:i+medmeanFeatures_shrinkingFactor],axis=1,keepdims=True)

    mitte_features_train = mitte_features_train_shrunk
    mitte_features_test = mitte_features_test_shrunk
    abweichung_features_train=abweichung_features_train_shrunk
    abweichung_features_test=abweichung_features_test_shrunk
 
  data_dict["X_train"] = np.append(mitte_features_train,abweichung_features_train,axis=1)
  data_dict["X_test"] = np.append(mitte_features_test,abweichung_features_test,axis=1)

  return data_dict


def biosppyECG(data_dict, **args):
  assert "X_train" in data_dict.keys()
  assert "X_test" in data_dict.keys()
  
  train_rpeaks = []
  train_templates = []
  train_heart_rate= []

  for ts in tqdm(data_dict["X_train"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    ret = ecg(signal=ts[:firstnan], sampling_rate=300,show=False)
    train_rpeaks.append(ret["rpeaks"])
    train_templates.append(ret["templates"])
    train_heart_rate.append(ret["heart_rate"])

  test_rpeaks = []
  test_templates = []
  test_heart_rate= []

  for ts in tqdm(data_dict["X_test"]):
    try:
      firstnan = np.where(np.isnan(ts))[0][0]
    except:
      firstnan = len(ts)
    ret = ecg(signal=ts[:firstnan], sampling_rate=300,show=False)
    test_rpeaks.append(ret["rpeaks"])
    test_templates.append(ret["templates"])
    test_heart_rate.append(ret["heart_rate"])

  #data_dict["train_rpeaks"] = train_rpeaks
  data_dict["heartbeat_templates_train"] = train_templates
  data_dict["heart_rate_train"] = train_heart_rate

  #data_dict["test_rpeaks"] = test_rpeaks
  data_dict["heartbeat_templates_test"] = test_templates
  data_dict["heart_rate_test"] = test_heart_rate

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


def NO_DISPLAY_savePred(data_dict, **args):
    assert "y_test" in data_dict.keys()
    
    y_predict = pd.DataFrame(data_dict["y_test"])
    y_predict.index.name = "id"
    y_predict.to_csv( r'predictions/y__'+data_dict["_current_function_information_"]["results_save_path"].replace("./cache/","")+'.csv', index = True, header = [ "y" ])
    
    return data_dict


def balanceStupid(data_dict, **args):
    assert "X_train" in data_dict.keys()
    assert "y_train" in data_dict.keys()
    
    yANDx = np.append(data_dict["y_train"], data_dict["X_train"],axis=1)
    np.random.shuffle(yANDx)

    classes = []
    for i in range(4):
      classes.append(yANDx[yANDx[:,0]==i])
    
    max_entries = max(list(map(lambda x: x.shape[0], classes)))

    newClasses = []

    for i,c in enumerate(classes):
      newClasses.append(c)
      datapoints = c.shape[0]
      while True:
        new_datapoints = newClasses[i].shape[0]
        newPoints = min(datapoints,max_entries-new_datapoints)
        if newPoints == 0:
          break
        else:
          newClasses[i] = np.append(newClasses[i],c[:newPoints],axis=0)

    yANDx = newClasses[0]
    for i in range(1,len(newClasses)):
      yANDx = np.append(yANDx, newClasses[i], axis=0)
    np.random.shuffle(yANDx)

    data_dict["y_train"] = yANDx[:,0:1].astype(np.intc)
    data_dict["X_train"] = yANDx[:,1:]

    if "X_val" in data_dict.keys() and "y_val" in data_dict.keys():
      yANDx = np.append(data_dict["y_val"], data_dict["X_val"],axis=1)
      np.random.shuffle(yANDx)

      classes = []
      for i in range(4):
        classes.append(yANDx[yANDx[:,0]==i])
      
      max_entries = max(list(map(lambda x: x.shape[0], classes)))

      newClasses = []

      for i,c in enumerate(classes):
        newClasses.append(c)
        datapoints = c.shape[0]
        while True:
          new_datapoints = newClasses[i].shape[0]
          newPoints = min(datapoints,max_entries-new_datapoints)
          if newPoints == 0:
            break
          else:
            newClasses[i] = np.append(newClasses[i],c[:newPoints],axis=0)

      yANDx = newClasses[0]
      for i in range(1,len(newClasses)):
        yANDx = np.append(yANDx, newClasses[i], axis=0)
      np.random.shuffle(yANDx)

      data_dict["y_val"] = yANDx[:,0:1].astype(np.intc)
      data_dict["X_val"] = yANDx[:,1:]
    else:
      print("Warning. Make sure you are creating a validation set before calling this method in case you want to use one.")


    return data_dict


def NO_DISPLAY_plotLosses(data_dict, **args):
  if "train_losses" in data_dict:
    plt.plot(range(len(data_dict["train_losses"])),data_dict["train_losses"], label="train loss")
    if "val_losses" in data_dict:
      plt.plot(range(len(data_dict["train_losses"])),data_dict["val_losses"], label="val loss")
      plt.title(f"hard F1 score on predicted validation set {f1_score(data_dict['y_val'], data_dict['y_val_predicted'],average='micro')}")
    plt.legend()
    plt.show()
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