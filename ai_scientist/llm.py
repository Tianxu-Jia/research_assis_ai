import backoff
import openai
import json
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import InferenceClient
from openai import OpenAI

huggingface_client = InferenceClient(    
    api_key="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",
)


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.75,
    n_responses=1,
):
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif "claude" in model:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)
    else:
        # TODO: This is only supported for GPT-4 in our reviewer pipeline.
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


"""
Retrieves a response from a large language model (LLM) using the provided message, client, model, and system message.

Args:
    msg (str): The input message to be sent to the LLM.
    client (object): The client object for the LLM service.
    model (str): The name of the LLM model to use.
    system_message (str): The system message to be included in the LLM request.
    print_debug (bool, optional): Whether to print debug information. Defaults to False.
    msg_history (list, optional): The message history to be included in the LLM request. Defaults to an empty list.
    temperature (float, optional): The temperature parameter for the LLM request. Defaults to 0.75.

Returns:
    tuple: A tuple containing the LLM response content and the updated message history.

###########
This function is designed to interact with different large language models (LLMs) and get a response from them. 
Its main purpose is to send a message to an LLM and receive a reply, 
handling the specifics of how to communicate with different types of models.

The function takes several inputs:

1. msg: The message to send to the LLM
2. client: An object used to communicate with the LLM service
3. model: The name of the LLM to use
4. system_message: A message that sets the context for the LLM
5. print_debug: A flag to enable or disable debug printing
6. msg_history: A list of previous messages in the conversation
7. temperature: A setting that affects how random or deterministic the LLM's responses are

The function produces two outputs:

1. content: The response from the LLM
2. new_msg_history: An updated list of messages including the new response

To achieve its purpose, the function first checks which type of model is being used. 
It then formats the input message and any previous conversation history in the way that specific model expects. 
It sends this formatted request to the LLM using the provided client object and receives a response. 
Finally, it extracts the text content from this response and updates the message history.

The function handles four different types of models:

1. Claude models
2. GPT-4 models (with specific version numbers)
3. A Deepseek Coder model
4. LLaMA models

For each model type, the function structures the request slightly differently. 
For example, Claude models use a different format for the message history compared to the other models. 
The GPT-4 models include a 'seed' parameter for reproducibility.

An important part of the logic is how it builds up the message history. 
Each time a message is sent or received, it's added to the history. 
This allows for context to be maintained across multiple interactions with the LLM.

If debug printing is enabled, the function will print out the entire conversation history and the LLM's response. 
This can be useful for understanding what's being sent to and received from the model.

The function uses a decorator (@backoff.on_exception) to automatically retry the request 
if it encounters rate limit errors or timeouts from the LLM service. 
This makes the function more robust to temporary issues with the LLM service.

Overall, this function serves as a flexible interface for interacting with various LLMs, 
handling the differences between them and providing a consistent way to send messages and receive responses.
"""
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(msg, client, model, system_message, print_debug=True, msg_history=None, temperature=0.75):
    if msg_history is None:
        msg_history = []
    
    
    #if model.split('/')[0] == "huggingface":
    #    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    #    tokenizer = AutoTokenizer.from_pretrained("/".join(model.split('/')[1:]), trust_remote_code=True)
    #    hf_model = AutoModelForCausalLM.from_pretrained("/".join(model.split('/')[1:]), torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    #    hf_model = hf_model.eval()
        
    #    content, history = hf_model.chat(tokenizer, query=system_message, history=new_msg_history, max_new_tokens=32768, temperature=0.5, top_p=0.7)
    #    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    #    debug = 1
    '''
    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=3000,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
     '''   
    if model.split("/")[0] == "anthropic":
        
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]

        client_model = model.split("/")[1]

        response = client.messages.create(
            model=client_model,
            max_tokens=3000,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )

        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
        debou = 1

    elif model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "llama3.1", "deepseek-coder-v2", "phi3.5",   ## ollama
        "THUDM/LongWriter-glm4-9b",   #HuggingFace
    ]:
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
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    """
    Extracts the JSON string between the specified start and end markers from the given LLM output.
    
    Args:
        llm_output (str): The output from the language model, which may contain a JSON string enclosed in markers.
    
    Returns:
        dict or None: The parsed JSON object if the markers and JSON string are found, otherwise None.
    """
    json_start_marker = "```json"
    json_end_marker = "```"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)  # Move past the marker
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None  # JSON markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    json_string = llm_output[start_index:end_index].strip()
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None  # Invalid JSON format
