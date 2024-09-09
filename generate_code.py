import json
import anthropic
import openai
import os.path as osp
import os
import shutil
import json
import sys
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datetime import datetime
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.utils import print_time, parse_arguments #, get_gpu_memory


# Open the JSON file
with open('./templates/2d_diffusion/ideas.json', 'r') as file:
    # Load the JSON data
    novel_ideas = json.load(file)

def gen_code(base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=True):

    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    destination_dir = osp.join(results_dir, idea_name)
    
    
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(destination_dir, "experiment.py")
    vis_file = osp.join(destination_dir, "plot.py")
    notes = osp.join(destination_dir, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(destination_dir, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{destination_dir}/{idea_name}_aider.txt"
        )
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
            success = perform_experiments(idea, destination_dir, coder, baseline_results)
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
            writeup_file = osp.join(destination_dir, "latex", "template.tex")
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
                perform_writeup(idea, destination_dir, coder, client, client_model)
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
                paper_text = load_paper(f"{destination_dir}/{idea['Name']}.pdf")
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
                with open(osp.join(destination_dir, "review.txt"), "w") as f:
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
                    coder, destination_dir, f"{destination_dir}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{destination_dir}/{idea['Name']}_improved.pdf")
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
                with open(osp.join(destination_dir, "review_improved.txt"), "w") as f:
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
    base_dir = 'templates/2d_diffusion'
    results_dir = 'results/2d_diffusion'
    model = 'anthropic/claude-3-5-sonnet-20240620'
    client_model = 'anthropic/claude-3-5-sonnet-20240620'
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    writeup = 'latex'
    improvement = False

    # do_idea(base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=True):
    for idea in novel_ideas:
        try:
            success = gen_code(
                base_dir,
                results_dir,
                idea,
                model,
                client,
                client_model,
                writeup,
                improvement,
                )
            print(f"Completed idea: {idea['Name']}, Success: {success}")
        except Exception as e:
            print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

        print("All ideas evaluated.")