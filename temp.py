from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="phi3.5",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)

debug = 1


response = client.chat.completions.create(
        model="llama2",
        messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
        )
        
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
'''
from huggingface_hub import InferenceClient

huggingface_client = InferenceClient(    
    api_key="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",
)

#new_msg_history = [{"role":"user", "content": "Hello"}] + [{"role": "user", "content": "world"}]
new_msg_history = []

response = huggingface_client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        #{"role": "user", "content": "Count to 10"},
        *new_msg_history,
    ],
    temperature=0.75,    
    n=1,
    stop=None,
    seed=0,
    stream=False,
    max_tokens=1024,
)
content = response.choices[0].message.content

print(content)

#for chunk in output:
#    print(chunk.choices[0].delta.content)

debug = 1
'''