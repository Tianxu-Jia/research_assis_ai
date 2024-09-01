
from semanticscholar import SemanticScholar
sch = SemanticScholar()
paper = sch.get_paper('10.1093/mind/lix.236.433')
paper.title

'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = "Count to 10"
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=32768, temperature=0.5, top_p=0.7)
print(response)

debug = 1


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


from openai import OpenAI
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="mistralai/mistral-large-2-instruct",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Count to 10"},
        ],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

content = completion.choices[0].message.content
print(content)

#for chunk in completion:
#  if chunk.choices[0].delta.content is not None:
#    print(chunk.choices[0].delta.content, end="")

debug = 1


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
        {"role": "user", "content": "Count to 10"},
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
'''
#for chunk in output:
#    print(chunk.choices[0].delta.content)

debug =1

'''
    elif model in ["llama3.1", "deepseek-coder-v2", "phi3.5"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
            seed=0,
        )
        
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]     
        
        
    '''