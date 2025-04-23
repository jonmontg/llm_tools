import os
import pandas as pd
import numpy as np
from pathlib import Path

def cache(path):
  return Cache(path)

class Cache:
  def __init__(self, path):
    self.path = path
    dirp = os.path.dirname(path)
    if not os.path.isdir(dirp):
      os.makedirs(dirp)

  def _abspath(self, *endpoint):
    return os.path.join(self.path, *endpoint)

  def check(self, *endpoint):
    return Path(self._abspath(*endpoint)).is_file()

  def save_cache_csv(self, df, *endpoint):
    df.to_csv(self._abspath(*endpoint), index=False)

  def read_cache_csv(self, *endpoint):
    path = self._abspath(*endpoint)
    print(f"Reading from cache: {str(path)}")
    return pd.read_csv(path)

  def save_cache_np(self, arr, *endpoint):
    path = self._abspath(*endpoint)
    np.savetxt(path, arr, delimiter=",")

  def read_cache_np(self, *endpoint, dtype=np.float32):
    path = self._abspath(*endpoint)
    return np.genfromtxt(path, delimiter=",", dtype=dtype)