from datetime import datetime
import argparse
import torch

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
            #"huggingface/THUDM/LongWriter-glm4-9b",
            # anthropic claude
            "anthropic/claude-3-5-sonnet-20240620"

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

