import pandas as pd
import json


class Pipeline:

  def __init__(self, stage, method_prev, method=None):

    self.stage = stage
    self.method_prev = method_prev
    self.method = f"s{ self.stage }" if method is None else method
    self.names = ["X_train", "y_train", "X_test"]

  def load(self):

    return [self._load(name) for name in self.names]

  def _load(self, name):

    file = f"stage{ self.stage - 1 }_out/{ name }_{ self.method_prev }.csv"
    return pd.read_csv(file)

  def save(self, X_train=None, y_train=None, X_test=None, method=None):

    self._save("X_train", X_train, method)
    self._save("y_train", y_train, method)
    self._save("X_test", X_test, method)

  #must be dataframe with header row
  def _save(self, name, df, method=None):

    if method is None:

      method = self.method

    if df is None:

      df = self._load(name)

    file = f"stage{ self.stage }_out/{ name }_{ self.method_prev }_{ method }.csv"
    df.to_csv(file, index=False)

  def save_prediction(self, y, method=None):

    if method is None:

      method = self.method

    y.index.name = "id"
    file = f"stage{ self.stage }_out/{ self.method_prev }_{ method }.csv"
    y.to_csv(file, index=True, header=["y"])

  def save_diagnostics(self, dictionary, method=None):

    if method is None:

      method = self.method

    file = f"stage{ self.stage }_out/_{ self.method_prev }_{ method }.json"

    with open(file, "w") as fp:

      json.dump(dictionary, fp, indent=4)
