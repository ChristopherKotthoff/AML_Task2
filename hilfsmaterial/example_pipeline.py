from pipeline import pipeline

def add(data_dict, hyp_add1, hyp_add2=0.01, **args):
  assert "a" in data_dict.keys()
  assert "b" in data_dict.keys()
  a = data_dict["a"]
  b = data_dict["b"]

  data_dict["z"] = (a+b)*10.0+hyp_add1+hyp_add2

  return data_dict


def multi(data_dict, hyp_multi1, hyp_multi2, **args):
  assert "z" in data_dict.keys()
  z = data_dict["z"]

  data_dict["multi_result"] = z*hyp_multi1*hyp_multi2

  return data_dict


hyperparameter_dictionary = {
  "hyp_add1":0.1,
  "hyp_add2":0.02,
  "hyp_multi1":3,
  "hyp_multi2":5
}

final_data_dict = pipeline([add, multi],hyperparameter_dictionary, use_cached_states=True, save_states_to_cache=True, load_example_instead_of_huge_X_files=True)
print(final_data_dict)
# kann auch so benutzt werden:
# pipeline([add, multi], hyperparameter_dictionary, load_example_instead_of_huge_X_files=True)
# pipeline([add, multi], hyperparameter_dictionary)
# die option load_example_instead_of_huge_X_files gibt es nur, damit hier auf repl.it nicht gleich die 300mb files geladen werden beim ausprobieren

