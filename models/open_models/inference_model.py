import re
from threading import Lock, Thread
from queue import Queue

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


model_paths = {
    "llama-3.1-8b": "/data1/sunq/projects/models/meta-llama/Llama-3.1-8B-Instruct/",
    "Qwen2.5-7B": "/data1/sunq/projects/models/Qwen/Qwen2.5-7B-Instruct/",
    "Ministral-8B-Instruct-2410": "/data1/sunq/projects/models/mistralai/Ministral-8B-Instruct-2410/",
}


class Args:
    def __init__(self, model_path, temperature=0.7, max_length=2048, top_p=0.9, n=5):
        self.model_path = model_path
        self.temperature = temperature
        self.max_length = max_length
        self.top_p = top_p
        self.n = n


class Model:
    def __init__(self, model_name="llama-2-7b", gpu_memory_utilization = 0.9):
        args = self.load_llama_params(model_name)

        # Detect available GPU count
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Detected {num_gpus} available GPUs")
        else:
            num_gpus = 0
            print("No GPU detected, will use CPU")

        # Set tensor parallel size
        tensor_parallel_size = num_gpus if num_gpus > 0 else 1
      
        print("gpu_memory_utilization: ", gpu_memory_utilization)
        self.model = LLM(model=args.model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, max_model_len= args.max_length)
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.stop_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.args = args
        self.lock = Lock()  # Add thread lock
        self.request_queue = Queue()  # Request queue
        self.start_processing()  # Start processing thread

    def get_model_name(self):
        return self.model_name
    
    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def load_llama_params(model_name="llama-2-7b"):
        model_path = model_paths.get(model_name)
        if not model_path:
            raise ValueError(f"Model {model_name} is not available. Please check the model name or path")
        return Args(model_path)

    def start_processing(self):
        """Start background thread to process requests in queue"""
        def process_queue():
            while True:
                prompt, response_queue, temperature, max_tokens, n, top_p, stop_symbol = self.request_queue.get()
                response = self._predict_internal(prompt, temperature, max_tokens, n, top_p, stop_symbol)
                response_queue.put(response)
                self.request_queue.task_done()

        Thread(target=process_queue, daemon=True).start()

    def predict(self, prompt, temperature=None, max_tokens=None, n=None, top_p=None, stop_symbol="End of answer."):
        """Submit request to queue and return response"""
        response_queue = Queue()
        self.request_queue.put((prompt, response_queue, temperature, max_tokens, n, top_p, stop_symbol))
        return response_queue.get()

    def _predict_internal(self, prompt, temperature=None, max_tokens=None, n=None, top_p=None, stop_symbol="End of answer."):
        # Use specified parameters or default parameters

        if isinstance(stop_symbol, str):
            stop_symbol = [stop_symbol]

        if stop_symbol is not None and stop_symbol != "":
            stop_symbol = stop_symbol + [self.tokenizer.eos_token, "End of answer."]
        else:
            stop_symbol = [self.tokenizer.eos_token, "End of answer."]


        sampling_params = SamplingParams(
            temperature=temperature or self.args.temperature,
            top_p= top_p or self.args.top_p,
            max_tokens=max_tokens or self.args.max_length,
            n=n or self.args.n,
            include_stop_str_in_output=True,
            stop=stop_symbol,
            ignore_eos=False,
            logprobs=20
        )

        with self.lock:  # Lock model instance to ensure thread safety
            outputs = self.model.generate([prompt], sampling_params)
            responses = []
            
            # Calculate prompt_tokens
            prompt_tokens = len(self.tokenizer.encode(prompt))
        
            # print("\n" + "@"*40)
            # print(outputs[0].outputs)
            # print("@"*40 + "\n")

            # Process and return results
            for output in outputs[0].outputs:
                response_text = output.text
                # Calculate completion_tokens
                completion_tokens = len(self.tokenizer.encode(response_text))

                if not sampling_params.include_stop_str_in_output:
                    end_pos = response_text.find(stop_symbol)
                    if end_pos != -1:
                        response_text = response_text[:end_pos]
                
                if re.search(r"End of answer\.", response_text):
                    response_text = re.split(r"End of answer\.", response_text)[0]
                matches = list(re.finditer(r"Question:", response_text))
                if len(matches) >= 2:
                    second_question_pos = matches[1].start()
                    response_text = response_text[:second_question_pos]
                
                response = {
                    "content": response_text.strip(),
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "output_token_ids": output.token_ids,
                    "cumulative_logprob": output.cumulative_logprob,
                    "logprobs": output.logprobs,
                }
                responses.append(response)
        
        return responses
