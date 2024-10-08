import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
from ai_scientist.prompts import coder_prompt, next_prompt_1, next_prompt_2, next_prompt_3
#from prompts import coder_prompt, next_prompt_1, next_prompt_2, next_prompt_3

MAX_ITERS = 4
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 1500


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=7200):
    cwd = osp.abspath(folder_name)
    # COPY CODE SO WE CAN SEE IT.
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"run_{run_num}.py"),
    )

    # LAUNCH COMMAND
    command = [
        "python", 
        "-Xfrozen_modules=off",
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            #next_prompt = next_prompt_1.format(results=results,
            #                                   run_num=run_num,)
            
            next_prompt = f"""Run {run_num} completed. Here are the results:
                            {results}

                            Decide if you need to re-plan your experiments given the result (you often will not need to).

                            Someone else will be using `notes.txt` to perform a writeup on this in the future.
                            Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

                            Then, implement the next thing on your list.
                            We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
                            YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
                            If you are finished with experiments, respond with 'ALL_COMPLETED'."""
            

        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt


# RUN PLOTTING
def run_plotting(folder_name, timeout=600):
    cwd = osp.abspath(folder_name)
    # LAUNCH COMMAND
    command = [
        "python",
        "-Xfrozen_modules=off",
        "plot.py",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Plotting failed with return code {result.returncode}")
            next_prompt = f"Plotting failed with the following error {result.stderr}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Plotting timed out after {timeout} seconds")
        next_prompt = f"Plotting timed out after {timeout} seconds"
        return 1, next_prompt


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        
        coder_out = coder.run(next_prompt)
        print(coder_out)
        if "ALL_COMPLETED" in coder_out:
            break
        
        return_code, next_prompt = run_experiment(folder_name, run)
        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Not all experiments completed.")
        return False

    current_iter = 0
    next_prompt = next_prompt_2 
    while True:
        coder_out = coder.run(next_prompt)
        return_code, next_prompt = run_plotting(folder_name)
        current_iter += 1
        if return_code == 0 or current_iter >= MAX_ITERS:
            break
    next_prompt = next_prompt_3 
    coder.run(next_prompt)

    return True

if __name__ == "__main__":
    # test run_experiment function
    folder_name = 'results/2d_diffusion/20240904_191601_learning_rate_schedule'
    run_num = 1
    timeout = 7200
    return_code, next_prompt = run_experiment(folder_name, run_num, timeout)
    print(return_code, next_prompt)

    debug = 1


