import pipeline
import numpy as np
from pipeline import pipeline
from chris import ldData, mlpClassification, biosppyRPeaks, biosppyExtract, plotNormHeartbeat,normalizeTemplates,medmeanFeatures,makeTrainValSet
from peak_detection_2_olin import peakDetecOlin
from heinrich import crop, inv

def custom(data_dict, **args):
    data_dict["X_train"] = data_dict["X_train"][885:888]
    data_dict["y_train"] = data_dict["y_train"][885:888]
    data_dict["X_test"] = data_dict["X_test"][:10]

    return data_dict




hyperparameter_dictionary = {"mlpClassification_epochs":20,
                            "mlpClassification_useValidationSet":True, 
                            "mlpClassification_makePrediction":True,
                            "plotNormHeartbeat_index":6,

                            "normalizeTemplates_normOverEntireTimeseries":False,
                            }

#final_data_dict = pipeline([ldData, custom, mlpClassification],hyperparameter_dictionary,save_states_to_cache=False)
final_data_dict = pipeline([ldData, crop, inv, custom, biosppyRPeaks, biosppyExtract, medmeanFeatures, makeTrainValSet, mlpClassification],hyperparameter_dictionary,save_states_to_cache=False)
print(final_data_dict)