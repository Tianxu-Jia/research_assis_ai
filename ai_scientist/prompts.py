idea_first_prompt = """{task_description}
    <experiment.py>
    {code}
    </experiment.py>

    Here are the ideas that you have already generated:

    '''
    {prev_ideas_string}
    '''

    Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
    Note that you will not have access to any additional resources or datasets.
    Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

    Respond in the following format:

    THOUGHT:
    <THOUGHT>

    NEW IDEA JSON:
    ```json
    <JSON>
    ```

    In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

    In <JSON>, provide the new idea in JSON format with the following fields:
    - "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
    - "Title": A title for the idea, will be used for the report writing.
    - "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
    - "Interestingness": A rating from 1 to 10 (lowest to highest).
    - "Feasibility": A rating from 1 to 10 (lowest to highest).
    - "Novelty": A rating from 1 to 10 (lowest to highest).

    Be cautious and realistic on your ratings.
    This JSON will be automatically parsed, so ensure the format is precise.
    You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list."""

next_prompt_1 = """Run {run_num} completed. Here are the results:
{results}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

Then, implement the next thing on your list.
We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with experiments, respond with 'ALL_COMPLETED'."""

next_prompt_2 = """
Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 

In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.

Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.

We will be running the command `python plot.py` to generate the plots.
"""

next_prompt_3 = """
Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth.

Somebody else will be using `notes.txt` to write a report on this in the future.
"""



