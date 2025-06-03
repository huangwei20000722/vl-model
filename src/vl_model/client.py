import os

import openai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class Client(object):
  def __init__(self):
    self.client = openai.OpenAI(
      api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )


client = Client()


def get_client():
  return client.client
