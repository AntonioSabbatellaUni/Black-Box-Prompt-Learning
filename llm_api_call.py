from openai import OpenAI
import string
import re
import datetime
import json
from dotenv import load_dotenv
import os

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# client = OpenAI(api_key='sk-h..' )
# client = OpenAI(base_url="https://d7be-81-56-46-24.ngrok-free.app/v1", api_key="not-needed")



def chat(system, user_message, max_tokens=5, temperature=0.0):
  assert isinstance(system, str), "`system` should be a string"
  assert isinstance(user_message, str), "`user_message` should be a string"
  
  messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
  try:
    chat_completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, 
                                            max_tokens=max_tokens, temperature=temperature)
    response= chat_completion.choices[0].message.content
  except:
    response = "Error / Exception during chat completion, input text is: " + user_message + "system is: " + system
    print("[LOG] Error / Exception during chat completion, input text is: ", user_message)
    chat_completion = chat_completion


  current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
  # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  with open("dump_call.json", "a") as file:
    dump_data = {
      "time": current_time,
      "response": response,
      "messages": messages
    }
    json.dump(dump_data, file)
    file.write('\n')

  return response, chat_completion




def chat_LLM(input_text: str, LABEL2ID_CONFIG: dict) -> dict:
  assert isinstance(input_text, str), "`input_text` should be a string"
  assert isinstance(LABEL2ID_CONFIG, dict), "`LABEL2ID_CONFIG` should be a dict"

  # Remove punctuation
  input_text = input_text.translate(str.maketrans("", "", string.punctuation))

  LABEL2ID_CONFIG = {label.lower().replace(" ", ""): value for label, value in LABEL2ID_CONFIG.items()}
  labels = list(LABEL2ID_CONFIG.keys())
  labels = [label.lower() for label in labels]

  response, chat_completion = chat(system="label can be " + str(labels) + ".  input text is: ",
                   user_message=input_text, max_tokens=15, temperature=0.0)
  response = response.lower()
  #Remove punctuation and non-alphanumeric characters
  response = re.sub(r'[^a-zA-Z0-9 ]', ' ', response)
  

  word_list = response.split()

  # label_occurrences = {label: response.count(label) for label in labels}
  label_occurrences = {label: word_list.count(label) for label in labels}
  # print("res: ",response)
  # print(label_occurrences)
  max_occurrence = max(label_occurrences.values())

  if max_occurrence == 0 or list(label_occurrences.values()).count(max_occurrence) > 1:
    return {"label": "-1", "response":chat_completion}

  selected_label = max(label_occurrences, key=label_occurrences.get)

  return {"label": LABEL2ID_CONFIG[selected_label], "response":chat_completion}






system_prompt = "label can be [Yes, No].  input text is: " 
user_message = "It also said it expects a civil complaint by the Securities and Exchange Commission.</s>Stewart also faces a separate in-vestigation by the Securities and Exchange Commis-sion.?<mask>"

# print(chat(system_prompt, user_message, max_tokens=5, temperature=0.0))
# print(chat_LLM(user_message, {"Yes": 1, "No": 0}))

system_prompt = "label can be [great, terrible].  input text is: "
user_message = "it 'll only put you to sleep. It was<mask>"

# response = chat(system_prompt, user_message, max_tokens=15, temperature=0.0)
# print(chat_LLM(user_message, {" terrible": 0, " great": 1}))