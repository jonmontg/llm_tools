import requests
import time
import json
import os

GPT_MODELS = {
  "gpt-3.5-turbo-0125": {
    "max_tokens_per_minute": 60000,
    "price_per_1m_input_tokens": 0.5,
    "price_per_1m_output_tokens": 1.5
  }
}

def supported_models():
  return list(GPT_MODELS.keys())

class GPTQueryManager:
  def __init__(self, model, api_key=None):
    if model not in GPT_MODELS:
      raise "GPT model is not defined in the GPT_MODELS constant."
    self.model = model
    self.api_key = api_key or os.environ["GPT_API_KEY"]

  def calculate_usage(self, response):
    input_tokens = json.loads(response.text)["usage"]["prompt_tokens"]
    output_tokens = json.loads(response.text)["usage"]["completion_tokens"]
    input_price = input_tokens*GPT_MODELS[self.model]["price_per_1m_input_tokens"]/1000000.0
    output_price = output_tokens*GPT_MODELS[self.model]["price_per_1m_input_tokens"]/1000000.0
    return input_tokens, output_tokens, input_price + output_price

  def query(self, prompt, print_usage=True):
    response = requests.post(
      "https://api.openai.com/v1/chat/completions",
      json={
        "model": self.model,
        "messages": [prompt],
      },
      headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
      }
    )
    if print_usage:
      input_tokens, output_tokens, price = self.calculate_usage(response)
      print(f"Input tokens={input_tokens}, Output tokens={output_tokens}, Price=${price}")
    return response

  def query_gpt_multi(self, *prompts, model="gpt-3.5-turbo-0125", print_usage=True):
    responses = []
    start_time = time.time()
    for prompt in prompts:
      response = self.query(prompt, model=model, api_key=self.api_key, print_usage=False)
      if response.get("error", "") == "Rate limit exceeded":
        time.sleep(60-((time.time()-start_time)%60))
        response = self.query(prompt, model=model, print_usage=False)
      responses.append(response)
    if print_usage:
      usages = list(map(self.calculate_usage, responses))
      input_tokens, output_tokens, price = [(sum(usage[0]), sum(usage[1]), sum(usage[2])) for usage in zip(*usages)]
      print(f"Input tokens={input_tokens}, Output tokens={output_tokens}, Price=${price}")
    return responses
