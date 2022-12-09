import pipeline
import numpy as np
from pipeline import pipeline
from chris import ldData, mlpClassification, biosppyRPeaks, biosppyExtract, NO_DISPLAY_plotNormHeartbeat,normalizeTemplates,medmeanFeatures,makeTrainValSet,NO_DISPLAY_savePred, biosppyECG,balanceStupid, NO_DISPLAY_plotLosses, mlpConvolution, medmeanFeaturesImproved, fuseXandMedMeans
from peak_detection_2_olin import peakDetecOlin
from heinrich import crop, inv, ecgExtract, rfClassification
from anova import anova
from olin_predictions import predictSimilarity
import matplotlib.pyplot as plt
from olin_utils import confMat
from olin import ecgExtractOlinFFT,ecgExtractOlinFFT26


def custom(data_dict, **args):
    try:
        firstnan = np.where(np.isnan(data_dict["X_train"][886]))[0][0]
    except:
        firstnan = len(data_dict["X_train"][886])

    usefulpart = data_dict["X_train"][886, 2000:firstnan].copy()
    data_dict["X_train"][886] = np.full(len(data_dict["X_train"][886]), np.nan, dtype=np.float32)
    data_dict["X_train"][886, 0:len(usefulpart)] = usefulpart

    # data_dict["X_train"] = data_dict["X_train"][885:888]
    # data_dict["y_train"] = data_dict["y_train"][885:888]
    # data_dict["X_test"] = data_dict["X_test"][:10]

    return data_dict


def xtoxbackup(data_dict, **args):
    assert "X_train" in data_dict.keys()
    assert "X_test" in data_dict.keys()

    data_dict["X_train_backup"] = data_dict["X_train"].copy()
    data_dict["X_test_backup"] = data_dict["X_test"].copy()

    return data_dict

def insertConvFeatures(data_dict, **args):
    assert "X_train" in data_dict.keys()
    assert "X_test" in data_dict.keys()

    train = np.loadtxt("x_train_conv_features.txt")
    test = np.loadtxt("x_test_conv_features.txt")

    data_dict["X_train"] = np.append(data_dict["X_train"],train, axis=1)
    data_dict["X_test"] = np.append(data_dict["X_test"],test, axis=1)

    return data_dict


hyper = {

    "inv_threshold": 0.6,
    "crop_location": 300,
    "mlpClassification_epochs": 200,
    "mlpClassification_useValidationSet": True,
    "mlpClassification_makePrediction": False,
    "makeTrainValSet_valPercent": 0.1,
    "rfClassification_depth": 4,
    "rfClassification_useValidationSet": False,
    "rfClassification_makePrediction": True,
    "anova_percentage": 0.7,
    "stackedClf_useValidationSet": True,
    "stackedClf_makePrediction": False,
    "used_cached_stats": True,
    "medmeanFeaturesImproved_putInMedMeans":True,
    "mlpConvolution_epochs": 100,
    "mlpConvolution_useValidationSet": False,
    "mlpConvolution_makePrediction": True,

}

# ecgExtractOlinFFT -> 20 fourier features
# ecgExtractOlinFFT35 -> 35 fourier features

# data = pipeline([ ldData, crop, inv, ecgExtractOlinC, rfClassification, NO_DISPLAY_savePred ], hyper, save_states_to_cache=True)
#final_data_dict = pipeline([ldData, crop, inv, ecgExtractOlinFFT, xtoxbackup, ldData, crop, inv, biosppyECG, medmeanFeaturesImproved, fuseXandMedMeans, makeTrainValSet,balanceStupid, mlpConvolution, NO_DISPLAY_plotLosses], hyper,save_states_to_cache=False)
#final_data_dict = pipeline([ldData, crop, inv, ecgExtractOlinFFT, xtoxbackup, ldData, crop, inv, biosppyECG, medmeanFeaturesImproved, fuseXandMedMeans, mlpConvolution, NO_DISPLAY_plotLosses], hyper,save_states_to_cache=False)




for i in range(2,50):
    hyper["rfClassification_depth"] = i
    print(f"now testing {i}")
    final_data_dict = pipeline([ldData,  crop, inv, ecgExtractOlinFFT26, rfClassification, NO_DISPLAY_savePred], hyper,save_states_to_cache=True)

if "y_val_hat" in final_data_dict:
  confMat(final_data_dict["y_val_hat"].reshape(-1), final_data_dict["y_val"].reshape(-1), visualize = True)



'''
hyper = {

    "inv_threshold": 0.6,
    "crop_location": 300,
    "mlpConvolution_epochs": 25,
    "mlpConvolution_useValidationSet": True,
    "mlpConvolution_makePrediction": True,
    "makeTrainValSet_valPercent": 0.1,
    "rfClassification_depth": 3,
    "rfClassification_useValidationSet": True,
    "rfClassification_makePrediction": False,
    "anova_percentage": 0.7,
    "mlpClassification_epochs":30,
    "mlpClassification_useValidationSet":True,
    "mlpClassification_makePrediction":True,
    "plotNormHeartbeat_index":6,
    "normalizeTemplates_normOverEntireTimeseries":False,
    "medmeanFeatures_shrinkingFactor":1,
}

final_data_dict = pipeline([ldData, crop, inv, biosppyECG, medmeanFeaturesImproved, makeTrainValSet, balanceStupid, mlpConvolution, NO_DISPLAY_plotLosses], hyper,save_states_to_cache=True)'''
#print("train losses")
#plt.plot(final_data_dict["train_losses"], color="blue")
#print("val losses")
#plt.plot(final_data_dict["val_losses"], color="green")

#final_data_dict = pipeline([ldData, custom, mlpClassification],hyperparameter_dictionary,save_states_to_cache=False)
#final_data_dict = pipeline([ldData, custom, crop, inv, peakDetecOlin, biosppyExtract, medmeanFeatures,makeTrainValSet,predictSimilarity,savePred],hyperparameter_dictionary,save_states_to_cache=True)
#final_data_dict = pipeline([ldData, custom, crop, inv, biosppyECG, medmeanFeatures,makeTrainValSet,balanceStupid,mlpClassification,NO_DISPLAY_savePred,NO_DISPLAY_plotLosses],hyperparameter_dictionary,save_states_to_cache=False)
#final_data_dict = pipeline([ldData, crop, inv, ecgExtract, anova,makeTrainValSet,balanceStupid,mlpClassification,NO_DISPLAY_savePred,NO_DISPLAY_plotLosses],hyper,save_states_to_cache=True)

if "y_val_predicted" in final_data_dict:
  confMat(final_data_dict["y_val_predicted"].reshape(-1), final_data_dict["y_val"].reshape(-1), visualize = True)

#final_data_dict = pipeline([ldData, crop, inv, biosppyRPeaks, biosppyExtract, medmeanFeatures,makeTrainValSet,mlpClassification,savePred],hyperparameter_dictionary,save_states_to_cache=True)
#print(final_data_dict)
