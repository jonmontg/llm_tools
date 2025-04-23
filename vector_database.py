import hashlib
import numpy as np
import pandas as pd
import faiss

class VectorDatabase:
  """
    A class to represent a vector database for document retrieval using embeddings.

    This class utilizes an embedding model to convert text documents into embeddings,
    stores them in a FAISS index for efficient similarity search, and supports caching
    of embeddings for faster repeated queries.

    Attributes:
        embedding_model (Embeddings): The embedding model used for text-to-vector conversion.
        cache (Cache): Cache object responsible for reading and saving embeddings
        data (pd.DataFrame): A DataFrame containing the documents and their corresponding embeddings.
        index (faiss.IndexFlatL2): A FAISS index for similarity search.
    """

  def __init__(self, documents, embedder, cache):
    """
    Initializes the VectorDatabase with a set of documents.

    Args:
        documents (list<str>): A DataFrame containing a 'document' column with text data.
        embedding_model (str): Name of the embedding model to use. Default is "text-embedding-3-small".
        cache (Cache): Cache object responsible for reading and saving embeddings
    """
    self.embedding_model = embedder
    self.cache = cache
    docs = list(documents)
    self.data = pd.DataFrame({
      "document": docs,
      "embedding": [self._get_embedding(doc) for doc in docs]
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
    cache_endpoint = ["embeddings", self.embedding_model.model, text_sha]
    if self.cache.check(*cache_endpoint):
      return self.cache.read_cache_np(*cache_endpoint)
    result = self.embedding_model.query(text, print_usage=True)
    if len(result) == 0:
      raise Exception(f"Invalid embedding array for {text}")
    self.cache.save_cache_np(result, *cache_endpoint)
    return result
