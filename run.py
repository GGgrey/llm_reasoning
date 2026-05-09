import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.multiprocessing as mp 
import torch
import json
import argparse
from tasks import get_task
from models.base_model import gpt_usage
from methods.test_time_compute import TestTimeCompute
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"


def run(args):
    args_dict = vars(args)
    logs, cnt_avg, cnt_any = [], 0, 0

    task = get_task(args.task)
    if args.task_end_index == -1:
        task_end_index = len(task)
    else:
        task_end_index = min(args.task_end_index, len(task))

    if args.baseline in ["greedy", "best_of_n", "mcts", "beam_search", "ToT", "weighted_majority"]:
        file = f'./logs/{args.task}/{args.baseline}/{args.backend}/{args.backend}_temp{args.temperature}_top-p{args.top_p}_{args.baseline}_{args.prompt_sample}_{args.method_evaluate}_sample_{args.n_generate_sample}_evaluate_{args.n_evaluate_sample}_start{args.task_start_index}_end{task_end_index}.json'
    elif args.baseline == 'self_refine':
        file = f'./logs/{args.task}/{args.baseline}/{args.backend}/{args.backend}_temp{args.temperature}_top-p{args.top_p}_{args.baseline}_{args.prompt_sample}_{args.method_evaluate}_num_iteration_{args.num_iteration}_sample_{args.n_generate_sample}_evaluate_{args.n_evaluate_sample}_start{args.task_start_index}_end{task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.baseline}/{args.backend}/{args.backend}_temp{args.temperature}_top-p{args.top_p}_{args.baseline}_{args.prompt_sample}_sample_{args.n_generate_sample}_evaluate_{args.n_evaluate_sample}_start{args.task_start_index}_end{task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # Resume from the last processed index
    if os.path.exists(file):
        try:
            with open(file, "r") as f:
                existing_logs = json.load(f)
      
            if not isinstance(existing_logs, list):
                raise ValueError("Log file format is invalid, expected a list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading log file: {e}")
            existing_logs = []

        processed_indices = [entry["idx"] for entry in existing_logs if "idx" in entry]
        print("\n" + '='*40)
        print(f"Resuming from the last processed index: {max(processed_indices)}")
        print('='*40 + "\n")
        if len(processed_indices) > 0:
            max_processed_index = max(processed_indices)

            if max_processed_index + 1 < task_end_index:
                args.task_start_index = max_processed_index + 1
                logs = existing_logs  
  
                completed_count = 0
                for entry in logs:
                    if "infos_output" in entry:
                        accs = [output["r"] for output in entry["infos_output"]]
                        cnt_avg += sum(accs) / len(accs)
                        cnt_any += any(accs)
                        completed_count += 1

                    if "usage_so_far" in entry:
                        usage_so_far = entry["usage_so_far"]
                        his_completion_tokens = usage_so_far["completion_tokens"]
                        his_prompt_tokens = usage_so_far["prompt_tokens"]
                        his_cost = usage_so_far["cost"]

            else:
                print("\n" + '='*40)
                print("All tasks have already been processed")
                print('='*40 + "\n")
                return
        else:
            completed_count = 0

    majority_result = []
    min_result = []
    average_result = []
    average_low_result = []
    dot_result = []
    perplexity_result = []
    total = []

    # Main loop
    solver = TestTimeCompute(task, args)
    for i in range(args.task_start_index, task_end_index):
        print(f"\n{'='*40}")
        print(f"Processing index {i}...")
        print(f"{'='*40}\n")

        x = task.get_input(i)
        ys, info = solver.solve(x, i, to_print=True)
        infos_output = [task.test_output(i, y) for y in ys]

        if args.task_start_index > 0:
            usage_so_far = gpt_usage(args.backend)
            completion_tokens = usage_so_far["completion_tokens"] + his_completion_tokens
            prompt_tokens = usage_so_far["prompt_tokens"] + his_prompt_tokens
            cost = usage_so_far["cost"] + his_cost
            overall_usage_so_far =    {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "cost": cost
            }
        else:
            overall_usage_so_far = gpt_usage(args.backend)
        
        if args.baseline == "mutual_consistency":
            mutual_completion_tokens = solver.get_mutual_completion_tokens()
            overall_usage_so_far["mutual_completion_tokens"] = mutual_completion_tokens
            overall_usage_so_far["completion_tokens"] += mutual_completion_tokens

        info.update({
            "idx": i,
            "ys": ys,
            "infos_output": infos_output,
            "usage_so_far": overall_usage_so_far,
            "args": args_dict  
        })
        logs.append(info)
        with open(file, "w") as f:
            json.dump(logs, f, indent=4)

        accs = [output["r"] for output in infos_output]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")

    processed_count = sum(1 for entry in logs if "infos_output" in entry)
    if processed_count > 0:
        final_log_info = {
            "average_accuracy": cnt_avg / processed_count,
            "any_correct": cnt_any / processed_count,
            "average_completion_tokens": overall_usage_so_far["completion_tokens"] / processed_count,
            "average_prompt_tokens": overall_usage_so_far["prompt_tokens"] / processed_count,
            "usage_so_far": overall_usage_so_far
        }
    else:
        final_log_info = {
            "average_accuracy": 0,
            "any_correct": 0,
            "usage_so_far": overall_usage_so_far
        }

        print("\n" + '='*40)
        print("Final statistics: ")
        print(f"Average accuracy: {final_log_info['average_accuracy']:.4f}")
        print(f"Any correct: {final_log_info['any_correct']:.4f}")
        print("Usage so far: ")
        print(f"Completion tokens: {final_log_info['usage_so_far']['completion_tokens']}")
        print(f"Prompt tokens: {final_log_info['usage_so_far']['prompt_tokens']}")
        print(f"Cost: ${final_log_info['usage_so_far']['cost']:.2f}")
        print('='*40 + "\n")

    logs.append(final_log_info)

    with open(file, "w") as f:
        json.dump(logs, f, indent=4)
    

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--backend", type=str, choices=["gpt-4", "gpt-3.5-turbo", "gpt-4o","llama-3.1-405b", "llama-3.1-70b", "llama-3.1-8b", "Qwen2.5-7B", "Qwen2.5-72B", "Mistral-7B-Instruct-v0.3", "Ministral-8B-Instruct-2410", "Qwen2.5-7B-self-tune", "Qwen2.5-7B-sft", "Qwen2.5-7B-dpo", "llama-3.1-8b-dpo", "QwQ-32B-Preview"], default="llama-3.1-8b", help="The name of the base model")
    parser.add_argument("--backend_prm", type=str, choices=["gpt-4", "gpt-3.5-turbo", "gpt-4o", "llama-3.1-405b", "llama-3.1-70b", "llama-3.1-8b", "Qwen2.5-7B", "Qwen2.5-72B", "Mistral-7B-Instruct-v0.3", "internlm2_5-step-prover-critic", "internlm2-1_8b-reward", "QwQ-32B-Preview"], default="internlm2-1_8b-reward", help="The name of the reward model")
    parser.add_argument("--inference_gpu_memory_utilization", type=float, default=0.75, help="The GPU memory utilization for inference model")
    parser.add_argument("--reward_gpu_memory_utilization", type=float, default=0.4, help="The GPU memory utilization for reward model")
    parser.add_argument("--port", type=int, default=8001, help="Port for the FastAPI service (default is 8001)")
    # Task
    parser.add_argument("--task", type=str, choices=["game24", "text", "crosswords", "bamboogle", "strategyqa", "hotpotqa", "gsm8k", "gsm8k_perb", "gsm_hard", "MATH500", "fever", "prontoqa", "humaneval"], default="gsm8k", help="The task name")
    parser.add_argument("--task_start_index", type=int, default=0)
    parser.add_argument("--task_end_index", type=int, default=-1)
    # Method
    parser.add_argument("--baseline", type=str, choices=["naive", "greedy", "majority", "best_of_n", "beam_search", "ToT_dfs", "self_refine", "mcts", "weighted_majority", "cer", "confidence_weighted_majority", "multi_llm_naive", "multi_llm_adaptive_majority", "multi_llm_adaptive_consensus", "adaptive_paraphrase_majority", "mutual_consistency"], default="mcts", help="The baseline name")
    parser.add_argument("--prompt_sample", type=str, choices=["standard", "cot", "reflect_cot"], default="cot", help="The prompt type")
    parser.add_argument("--method_evaluate", type=str, choices=["value", "vote", "random", "self_process_value", "self_result_value", "llm_as_binary", "llm_as_process_reward", "llm_as_reuslt_reward", "llm_as_reward_value", "llm_as_critic_value", "qwq_as_process_reward"], default="value")
    # Inference
    parser.add_argument("--temperature", type=float, default=0.7, help="The sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="The top-p value for sampling")
    parser.add_argument("--n_generate_sample", type=int, default=36, help="The number of samples to be generated")
    parser.add_argument("--n_evaluate_sample", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048, help="The max tokens limit")
    # Self-Refine
    parser.add_argument("--num_iteration", type=int, default=5, help="Number of iterations for self refine")
    # MCTS
    parser.add_argument("--num_simulation", type=int, default=5, help="Number of simulations")
    parser.add_argument("--sample_action", type=bool, default=False, help="Sample action flag")
    parser.add_argument("--c_base", type=int, default=19652, help="C base parameter")
    parser.add_argument("--c_puct", type=float, default=1.25, help="C puct parameter")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum depth for dfs search algorithms")  # Also for ToT
    # Beam Search
    parser.add_argument("--n_select_sample", type=int, default=2)
    parser.add_argument("--score_criterion", type=str, default='max')  # Also for ToT
    # ToT
    parser.add_argument("--method_generate", type=str, choices=['sample', 'propose'])
    parser.add_argument("--method_select", type=str, choices=['sample', 'greedy'], default='')
    parser.add_argument("--value_thresh", type=float, default=0.3, help="Value threshold for pruning in dfs search algorithms")
    parser.add_argument("--prune_ratio", type=float, default=0.4, help="Prune ratio for dfs search algorithms")
    parser.add_argument("--num_paths", type=int, default=3, help="Number of paths to explore in df search algorithms")
    # Multi-LLM Naive and Multi-LLM Majority
    parser.add_argument("--model_2nd", type=str, default="Qwen2.5-7B", help="The second model to be used")
    parser.add_argument("--inference_gpu_memory_utilization_2nd", type=float, default=0.3, help="The GPU memory utilization for the second model")
    parser.add_argument("--model_3rd", type=str, default="Ministral-8B-Instruct-2410", help="The third model to be used")
    parser.add_argument("--inference_gpu_memory_utilization_3rd", type=float, default=0.35, help="The GPU memory utilization for the third model")
    parser.add_argument("--conf_thresh", type=float, default=0.95, help="Confidence threshold for multi-LLM search")
    parser.add_argument("--mode", type=str, choices=["step_level_solitaire", "confidence_boost"], default="confidence_boost", help="The mode of multi-llm interaction")
    parser.add_argument("--window_size", type=int, default=5, help="The window size for repeated sampling, it should be set to be divided by the sample size")  # Also for Adaptive Methods
    # Confidence Weighted Majority
    parser.add_argument("--confidence_type", type=str, choices=["min", "average", "average_low", "std", "std_low", "dot", "average_entropy", "max_entropy", "perplexity", "response_improbability", "all"], default="all", help="Confidence score caculation methods")
    # Mutual Consistency
    parser.add_argument("--mutual_model", type=str, default="Qwen2.5-7B", help="The mutual reasoning model")
    parser.add_argument("--inference_gpu_memory_utilization_mutual", type=float, default=0.5, help="The GPU memory utilization for the second model")
    parser.add_argument("--mutual_iteration", type=int, default=5, help="Number of iterations for mutual reasoning")
    # CER
    parser.add_argument("--aggregate", type=bool, default=True, help="A boolean flag indicating whether to aggregate paths (True) or select the best path (False)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    torch.cuda.init()
    mp.set_start_method("spawn") 
    args = parse_args()
    print(args)
    run(args)
