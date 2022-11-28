import pipeline
import numpy as np
from pipeline import pipeline
from chris import ldData, mlpClassification
from peak_detection_2_olin import pipeline_stage_detect_peaks

'''def custom(data_dict, **args):
    x = data_dict["X_train"][:3000,:30]
    y = data_dict["y_train"][:3000]
    
    data_dict["X_test"] = data_dict["X_test"][:,:30]

    data_dict["X_val"] = data_dict["X_train"][3000:,:30]
    data_dict["y_val"] = data_dict["y_train"][3000:]

    data_dict["X_train"] = x
    data_dict["y_train"] = y



    return data_dict'''




def customloadolin(data_dict, **args):
    # Lets execute this code!

    # Load the data
    train_X_file = open("original_data/X_train.csv", 'r')
    train_X_str = train_X_file.readlines()
    train_X = [(np.asarray([int(x) for x in (line.split(","))])) for line in train_X_str[1:]]  # first line has "feature names"

    test_X_file = open("original_data/X_test.csv", 'r')
    test_X_str = test_X_file.readlines()
    test_X = [(np.asarray([int(x) for x in (line.split(","))])) for line in test_X_str[1:]]  # first line has "feature names"

    # create dictionary to feed into the pipeline
    data_dict = {}
    data_dict["X_train"] = train_X#[:5]
    data_dict["X_test"] = test_X#[:5]

    data_dict["y_train"] = np.loadtxt("./original_data/y_train.csv",
                               dtype=np.intc,
                               delimiter=',',
                               skiprows=1)[:, 1:]



    return data_dict















hyperparameter_dictionary = {"mlpClassification_epochs":20,
                            "mlpClassification_useValidationSet":True, 
                            "mlpClassification_makePrediction":True,
                            }

#final_data_dict = pipeline([ldData, custom, mlpClassification],hyperparameter_dictionary,save_states_to_cache=False)
final_data_dict = pipeline([customloadolin, pipeline_stage_detect_peaks],hyperparameter_dictionary,save_states_to_cache=True)
#print(final_data_dict)