import pipeline
import numpy as np
from pipeline import pipeline
from chris import ldData, mlpClassification, biosppyRPeaks, biosppyExtract, plotNormHeartbeat,normalizeTemplates,medmeanFeatures,makeTrainValSet,savePred
from peak_detection_2_olin import peakDetecOlin
from heinrich import crop, inv
from olin_predictions import predictSimilarity

def custom(data_dict, **args):

    try:
      firstnan = np.where(np.isnan(data_dict["X_train"][886]))[0][0]
    except:
      firstnan = len(data_dict["X_train"][886])

    usefulpart = data_dict["X_train"][886,2000:firstnan].copy()
    data_dict["X_train"][886] = np.full(len(data_dict["X_train"][886]), np.nan, dtype=np.float32)
    data_dict["X_train"][886,0:len(usefulpart)] = usefulpart
    
    #data_dict["X_train"] = data_dict["X_train"][885:888]
    #data_dict["y_train"] = data_dict["y_train"][885:888]
    #data_dict["X_test"] = data_dict["X_test"][:10]


    return data_dict




hyperparameter_dictionary = {"mlpClassification_epochs":20,
                            "mlpClassification_useValidationSet":True, 
                            "mlpClassification_makePrediction":True,
                            "plotNormHeartbeat_index":6,

                            "normalizeTemplates_normOverEntireTimeseries":False,
                            }

#final_data_dict = pipeline([ldData, custom, mlpClassification],hyperparameter_dictionary,save_states_to_cache=False)
final_data_dict = pipeline([ldData, custom, crop, inv, peakDetecOlin, biosppyExtract, medmeanFeatures,makeTrainValSet,predictSimilarity,savePred],hyperparameter_dictionary,save_states_to_cache=True)
#final_data_dict = pipeline([ldData, crop, inv, biosppyRPeaks, biosppyExtract, medmeanFeatures,makeTrainValSet,mlpClassification,savePred],hyperparameter_dictionary,save_states_to_cache=True)
#print(final_data_dict)
