import requests
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
import numpy as np

# Use the batch API for an asynchronous response for 50% cost
GPT_EMBEDDING_MODELS = {
  "text-embedding-3-small": {
    "price_per_1m_tokens": 0.02
  },
  "text-embedding-3-large": {
    "price_per_1m_tokens": 0.13
  }
}

def get_embedding_model(model, api_key=None):
  if model == "unixcoder-base":
    return UnixcoderBaseEmbeddings()
  elif model in GPT_EMBEDDING_MODELS:
    return GPTEmbeddings(model, api_key)
  else:
    raise Exception(f"Unsupported embedding model: {model}")

class Embeddings():
  def query(self, prompt):
    raise Exception("Not Implemented")

class UnixcoderBaseEmbeddings(Embeddings):
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    self.embedding_model = AutoModel.from_pretrained("microsoft/unixcoder-base")
    self.embedding_model.eval()
    self.model = "unixcoder-base"

  def query(self, prompt):
    with torch.no_grad():
      return (
        self.embedding_model(
          **self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        ).last_hidden_state[:, 0, :]
         .squeeze().numpy()
      )

class GPTEmbeddings(Embeddings):
  def __init__(self, model, api_key=None, print_usage=True):
    if model not in GPT_EMBEDDING_MODELS:
      raise "Embedding model is not defined in the GPT_EMBEDDING_MODELS constant."
    self.model = model
    self.api_key = api_key or os.environ["GPT_API_KEY"]
    self.print_usage = print_usage

  def calculate_usage(self, response):
    tokens = json.loads(response.text)["usage"]["total_tokens"]
    return tokens, tokens*GPT_EMBEDDING_MODELS[self.model]["price_per_1m_tokens"]/1000000.0

  def query(self, prompt):
    response = requests.post(
      "https://api.openai.com/v1/embeddings",
      json={
        "model": self.model,
        "input": prompt,
      },
      headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
      }
    )
    if response.status_code != 200:
      raise Exception(f"Request failed: {response.text}")
    if self.print_usage:
      input_tokens, price = self.calculate_usage(response)
      print(f"Input tokens={input_tokens}, Price=${price}")
    return np.array(json.loads(response.text)["data"][0]["embedding"], dtype=np.float32)
