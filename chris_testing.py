import pipeline
import numpy as np
from pipeline import pipeline
from chris import ldData, mlpClassification, biosppyRPeaks
from peak_detection_2_olin import peakDetecOlin

'''def custom(data_dict, **args):
    x = data_dict["X_train"][:3000,:30]
    y = data_dict["y_train"][:3000]
    
    data_dict["X_test"] = data_dict["X_test"][:,:30]

    data_dict["X_val"] = data_dict["X_train"][3000:,:30]
    data_dict["y_val"] = data_dict["y_train"][3000:]

    data_dict["X_train"] = x
    data_dict["y_train"] = y



    return data_dict'''


hyperparameter_dictionary = {"mlpClassification_epochs":20,
                            "mlpClassification_useValidationSet":True, 
                            "mlpClassification_makePrediction":True,
                            }

#final_data_dict = pipeline([ldData, custom, mlpClassification],hyperparameter_dictionary,save_states_to_cache=False)
final_data_dict = pipeline([ldData, peakDetecOlin],hyperparameter_dictionary,save_states_to_cache=True)
#print(final_data_dict)