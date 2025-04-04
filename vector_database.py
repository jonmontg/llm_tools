from pathlib import Path
import os
import hashlib
import numpy as np
import pandas as pd
import faiss

def _ensure_cache(path):
  dirp = os.path.dirname(path)
  if not os.path.isdir(dirp):
    os.makedirs(dirp)

class VectorDatabase:
  """
    A class to represent a vector database for document retrieval using embeddings.

    This class utilizes an embedding model to convert text documents into embeddings,
    stores them in a FAISS index for efficient similarity search, and supports caching
    of embeddings for faster repeated queries.

    Attributes:
        embedding_model (Embeddings): The embedding model used for text-to-vector conversion.
        cache_path (str): The path to store cached embeddings.
        data (pd.DataFrame): A DataFrame containing the documents and their corresponding embeddings.
        index (faiss.IndexFlatL2): A FAISS index for similarity search.
    """

  def __init__(self, documents, embedder, cache_path="."):
    """
    Initializes the VectorDatabase with a set of documents.

    Args:
        documents (list<str>): A DataFrame containing a 'document' column with text data.
        embedding_model (str): Name of the embedding model to use. Default is "text-embedding-3-small".
        cache_path (str): Path to store cached embeddings. Default is the current directory.
    """
    self.embedding_model = embedder
    self.cache_path = cache_path
    self.data = pd.DataFrame({
      "document": documents,
      "embedding": [self._get_embedding(doc) for doc in documents]
    })
    embeddings = np.vstack(self.data["embedding"])
    self.index = faiss.IndexFlatL2(embeddings.shape[1])
    self.index.add(embeddings)

  def search(self, text, n):
    """
    Searches the vector database for the top-N most similar documents to the input text.

    Args:
        text (str): The query text to search for.
        n (int): The number of top similar documents to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top-N similar documents.
    """
    embedded_query = self._get_embedding(text)
    d, i = self.index.search(np.array([embedded_query]), n)
    return self.data.iloc[[i[0][j] for j in range(n)]], d[0]

  def _get_embedding(self, text):
    """
    Generates or retrieves the embedding for a given text.

    Args:
        text (str): The text to embed.

    Returns:
        np.ndarray: The embedding vector for the input text.
    """
    text_sha = hashlib.sha1(text.encode()).hexdigest()
    cache_path = Path(self.cache_path, "cache", "embeddings", self.embedding_model.model, text_sha)
    if cache_path.is_file():
      return self._read_cache_np(cache_path)
    result = self.embedding_model.query(text, print_usage=True)
    self._save_cache_np(result, cache_path)
    return result

  def _save_cache_np(self, arr, path):
    """
    Saves a NumPy array to the cache.

    Args:
        arr (np.ndarray): The array to save.
        path (Path): The file path where the array will be stored.
    """
    _ensure_cache(path)
    if len(arr) == 0:
      raise Exception(f"Invalid embedding array: {path}")
    np.savetxt(path, arr, delimiter=",")

  def _read_cache_np(self, path, dtype=np.float32):
    """
    Reads a NumPy array from the cache.

    Args:
        path (Path): The file path to read the array from.
        dtype (np.dtype): The data type of the array. Default is np.float32.

    Returns:
        np.ndarray: The loaded array.
    """
    with open(path, "r") as file:
      contents = file.read()
    return np.genfromtxt(path, delimiter=",", dtype=dtype)
