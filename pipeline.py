import numpy as np
import inspect
from os.path import exists
import os
import pickle
from collections.abc import Mapping

# static
if not os.path.exists("./cache"):
  os.makedirs("./cache")


def _save_dict(path, dict):
  with open(path + ".pkl", 'wb') as f:
    pickle.dump(dict, f)


def _load_dict(path):
  with open(path + ".pkl", 'rb') as f:
    _dict = pickle.load(f)
  return _dict

def pipeline(functionlist,
             hyperparameter_dictionary,
             use_cached_states=True,
             save_states_to_cache=True):
  assert isinstance(functionlist, list)
  assert isinstance(hyperparameter_dictionary, Mapping) #test if dictionary

  function_information = []

  path = "./cache/"

  for pos_index, f in enumerate(functionlist):
    if str(inspect.signature(f))[-7:] != "**args)" or str(
        inspect.signature(f))[:11] != "(data_dict,":
      raise Exception(
        f"\n\n[Pipeline] Function header is wrong. It has to be like this:\n[Pipeline] def {f.__name__}(data_dict, ..., **args):\n[Pipeline] where \"...\" are the non-data hyperparameters that you use for your method."
      )

    hyperparameters = str(inspect.signature(f))[1:-1].split(", ")[1:-1]
    hyperparameters_values = []
    for h in hyperparameters:
      param_name = h.split("=")[0]
      if param_name in hyperparameter_dictionary.keys():
        hyperparameters_values.append(hyperparameter_dictionary[param_name])
      elif h.__contains__("="):
        hyperparameters_values.append(h.split("=")[1])
      else:
        raise Exception(
          f"\n\n[Pipeline] No parameter in hyperparameter_dictionary for parameter {param_name} in method {f.__name__}. Also no default value set in function header."
        )

    if pos_index != 0:
      path+="_"
    
    
    function_call_string = f.__name__+"("
    for j, v in enumerate(hyperparameters_values):
      function_call_string += str(v)
      if j != len(hyperparameters_values)-1:
        function_call_string+=","
    function_call_string += ")"

    path+=function_call_string

    dict = {
      "function": f,
      "function_name": f.__name__,
      "hyperparameters": str(inspect.signature(f))[1:-1].split(", ")[1:-1],
      "hyperparameters_values": hyperparameters_values,
      "results_save_path": path,
      "function_call_string": function_call_string
    }
    function_information.append(dict)

  next_function = 0
  if use_cached_states:
    # find state
    if exists(function_information[len(function_information)-1]["results_save_path"]+".pkl"):
      print(f"\n\n[Pipeline] This is no new run. It already exists at location {function_information[len(function_information)-1]['results_save_path']+'.pkl'}. Returning data_dict anyways.")
      return _load_dict(function_information[len(function_information)-1]["results_save_path"])
    
    for i in reversed(range(len(function_information)-1)):
      if exists(function_information[i]["results_save_path"]+".pkl"):
        data_dict = _load_dict(function_information[i]["results_save_path"])
        next_function = i + 1
        print(f"[Pipeline] Saved state found: {function_information[i]['results_save_path']}\n[Pipeline] Starting from function: {function_information[next_function]['function_name']}"
        )
        break

  if next_function == 0:
    if use_cached_states:
      print("[Pipeline] No saved state found. Starting from beginning")

      data_dict = {
        "a": 1,
        "b": 2,
        "c": 3,
      }


  for i in range(next_function, len(function_information)):
    print(f"[Pipeline] executing: {function_information[i]['function_call_string']}")
    data_dict = function_information[i]["function"](data_dict, **hyperparameter_dictionary)
    if not isinstance(data_dict, Mapping):
      raise Exception("\n\n[Pipeline] "+
        function_information[i]["function_name"] +
        " did not return the data_dict. Please have a look and make sure the updated data_dict is returned."
      )
    if save_states_to_cache:
      _save_dict(function_information[i]["results_save_path"], data_dict)
  return data_dict
