import openai
import os.path as osp
import shutil
import json
import argparse
import multiprocessing
import torch
import os
import time
import sys
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datetime import datetime
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement

NUM_REFLECTIONS = 3

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        choices=[
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
            # Anthropic Claude models via Amazon Bedrock
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-opus-20240229-v1:0"
            # Anthropic Claude models Vertex AI
            "vertex_ai/claude-3-opus@20240229",
            "vertex_ai/claude-3-5-sonnet@20240620",
            "vertex_ai/claude-3-sonnet@20240229",
            "vertex_ai/claude-3-haiku@20240307",
            # Ollamaa models
            "ollama/llama3.1",
            "ollama/deepseek-coder-v2:16b",
            "ollama/phi3.5",
            # huggingface models
            "huggingface/THUDM/LongWriter-glm4-9b",
        ],
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    return parser.parse_args()

def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))

def worker(queue, base_dir, results_dir, model, client, client_model, writeup, improvement, gpu_id):
    pass

''''
The do_idea function is designed to execute and evaluate a scientific idea using AI models. Here's a simple explanation of what it does:

This function takes several inputs, including information about the idea, directories for storing results, model details, 
and configuration options. Its purpose is to run an experiment based on the given idea, write up the results, review the paper, 
and potentially improve it.

The function starts by creating a new folder for the project, copying necessary files, and setting up logging. 
It then performs the following main steps:

1. It runs experiments related to the idea using AI models. This involves creating a "coder" object that can interact 
    with the AI model and execute the experiments.

2. If the experiments are successful, it moves on to writing up the results. Currently, 
    it supports writing in LaTeX format. The function uses the AI model to generate a scientific paper based on the experimental results.

3. After the writeup, it reviews the generated paper. This involves using another AI model (specifically GPT-4) to 
    analyze the paper and provide feedback.

4. If the "improvement" option is enabled, the function will attempt to improve the 
    paper based on the review feedback. It then generates a new, improved version of the paper and reviews it again.

Throughout this process, the function handles various file operations, such as creating directories, writing log files, 
and saving results. It also manages different AI models depending on the input parameters.

The main logic flow involves a series of steps (experiments, writeup, review, improvement) 
that are executed in order. If any step fails, the function typically logs the error and may return early.

In terms of output, this function produces several files:

1. A LaTeX paper summarizing the experiment and results
2. Log files documenting the process
3. Review files containing AI-generated feedback on the paper
4. If improvement is enabled, an improved version of the paper

The function returns True if all steps complete successfully, and False if there are any major errors.

This code represents a complex workflow for automating scientific research using AI, 
from running experiments to writing and refining papers. It's designed to handle different AI models 
and experimental setups, making it a flexible tool for AI-assisted scientific exploration.
'''
def do_idea(base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=True):

    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        print(f"*Starting Writeup*")
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(idea, folder_name, coder, client, client_model)
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                return False
            print("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()
        print(f"*Starting Review*")
        ## REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                print(f"Failed to perform review: {e}")
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            print_time()
            print(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                print(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()



if __name__ == "__main__":
    args = parse_arguments()
    
    """
    Checks the available GPUs and adjusts the number of parallel processes if necessary.
    
    This code block is responsible for the following:
    1. Retrieves the available GPUs using the `get_available_gpus()` function, passing the `args.gpus` parameter.
    2. Compares the number of requested parallel processes (`args.parallel`) to the number of available GPUs.
    3. If the requested parallel processes exceed the available GPUs, it prints a warning message and adjusts the `args.parallel` value to match the number of available GPUs.
    4. Prints the list of GPUs that will be used for the parallel processes.
    """
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")



    """
    Creates an AI client based on the specified model. Supports various AI models and providers, 
    including Anthropic's Claude, Amazon Bedrock, Google Vertex AI, and OpenAI's GPT-4 and DeepSeek Coder.
    
    The function checks the `args.model` parameter to determine the appropriate AI client and model to use. 
    It then prints a message indicating which client and model are being used.
    
    If the specified model is not supported, the function raises a `ValueError`.
    """
    # Create client
    if args.model.split('/')[0]== "huggingface":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"Using HuggingFace python API with model {args.model}.")

        client_model = args.model, #"/".join(args.model.split('/')[1:])
        client = AutoModelForCausalLM.from_pretrained("/".join(args.model.split('/')[1:]), torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        client = client.eval()

    elif args.model.split('/')[0]== "ollama":
        import ollama
        from ollama import Client

        print(f"Using Ollama python API with model {args.model}.")
        client_model = args.model
        client = Client(host='http://localhost:11434')

    elif args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {args.model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Vertex AI with model {client_model}.")
        client = anthropic.AnthropicVertex()
    elif args.model in ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18"]:
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        #client_model = "gpt-4o-2024-05-13"
        client_model = args.model
        client = openai.OpenAI()
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif args.model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")
    
    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )

    with open(osp.join(base_dir, "ideas_no_check_novelty.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    ideas = check_idea_novelty(
        ideas,
        base_dir=base_dir,
        client=client,
        model=client_model,
    )
    
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    

    #with open(osp.join(base_dir, "ideas.json"), 'r') as file:
    #    ideas = json.load(file)

    novel_ideas = [idea for idea in ideas if idea["novel"]]

    #######################################
    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                    gpu_id,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as e:
                print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

    print("All ideas evaluated.")

    debug_mode = True