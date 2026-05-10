import re
import os
import json

import sympy
import pandas as pd
from fuzzywuzzy import fuzz
import jsonlines

from tasks.base import Task, DATA_PATH
from prompts.gsm8k import * 


class GSM8K(Task):
    """
    Input (x)   : A question comparing the number of people related to Genghis Khan and Julius Caesar
    Output (y)  : A logical reasoning process leading to the answer
    Reward (r)  : 0 or 1, depending on whether the reasoning is correct
    Input Example: 
        Are more people today related to Genghis Khan than Julius Caesar?
    Output Example: 
        1. Determine the descendants of Genghis Khan (many modern individuals have DNA traced to him).
        2. Determine the known descendants of Julius Caesar (limited, mainly historical).
        3. Compare the two numbers: Genghis Khan's descendants significantly outnumber Caesar's.
        Final Answer: Yes, more people today are related to Genghis Khan than to Julius Caesar.
    """   
    def __init__(self, file="gsm8k.json"):
        super().__init__()
        path = os.path.join(DATA_PATH, "gsm8k", file)
        if not os.path.isfile(path):
            raise ValueError(f"File {file} not found at {path}")
        
        self.data = []
        self.ground_truth = []

        # Load data from JSON file
        with open(path) as f:
            json_data = json.load(f)
            for item in json_data:
                question = item.get("question")
                answer = item.get("answer")
                self.data.append(question)
                self.ground_truth.append(answer)

        
        self.value_cache = {}
        self.steps = 3
        self.stops = [".", ".", "End of answer."]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for data of length {len(self.data)}")
        return self.data[idx]

    def test_output(self, idx: int, output: str) -> dict:
        # Ground truth
        ground_truth = float(self.ground_truth[idx])
        tolerance = 0.01

        expression = self.extract_answer(output).replace(": ", "").strip()

        expression = expression.replace(",", "")

        match = re.search(r"[-+]?\d+(?:\.\d+)?", expression)
        if match:
            number_str = match.group(0)
            try:
                number = float(number_str)
                print(f"====Extracted====[{number}]")
                extracted_numbers = [number]
            except ValueError:
                print("====Error====: float conversion failed for extracted number")
                extracted_numbers = [0]
        else:
            print("====Error====: no number found in the extracted expression")
            extracted_numbers = [0]

        print(f"====GT===={ground_truth}====Extracted===={extracted_numbers}")

        for num in extracted_numbers:
            if abs(num - ground_truth) <= tolerance:
                return {"r": 1}

        return {"r": 0}

    def extract_answer(self, output: str) -> str:
        output = output.replace("\n", " ").strip()
        
        if "the final answer is" in output.lower():
            return output.lower().split("the final answer is")[-1].strip().split(".")[0].strip()
        elif "final answer" in output.lower():
            return output.lower().split("final answer")[-1].strip().split(".")[0].strip()
        elif "final refined solution" in output.lower():
            return output.lower().split("final refined solution")[-1].strip().split(".")[0].strip()
        elif "refined solution" in output.lower():
            return output.lower().split("refined solution")[-1].strip().split(".")[0].strip()
        else:
            return output.strip().split(".")[-1].strip()

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def reflect_cot_prompt_wrap(x: str, y: str = "") -> str:
        return reflect_cot_prompt.format(input=x) + y

    @staticmethod
    def paraphrase_question_prompt_wrap(x: str, y: str = "") -> str:
        return paraphrase_prompt.format(input=x)
    
    @staticmethod
    def progressive_question_prompt_wrap(x: str, y: str = "") -> str:
        return progressive_promot.format(input=x)

    @staticmethod
    def agent_cot_prompt_wrap(x: str, y: str = "", step: int = 1,knowledge: str = "") -> str:
        if step > 1:
            return agent_cot_prompt.format(input=x) + "\n" + y + "End of step." + "\nShared information: "+ knowledge
        else:
            return agent_cot_prompt.format(input=x) + "\n" + y + "End of step."

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        if "the final answer is" not in y.lower():
            return value_evaluate + x + "\nThought Process: " + y + "\nEvaluation Process:\n"
        else:
            return final_evaluate + x + "\n" + y + "\nEvaluation Process: \n"
        
    @staticmethod
    def self_process_value_prompt_wrap(x: str, y: str) -> str:
        return value_evaluate + x + "\nThought Process: " + y + "\nEvaluation Process:\n"

    @staticmethod
    def self_result_value_prompt_wrap(x: str, y: str) -> str:
        return final_evaluate + x + "\nSo the final answer is:" + y + "\nEvaluation Process: \n"  
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # Value map for scoring with probabilities between 0 and 1
        value_map = {"Impossible": 0.0, "Likely": 0.5, "Sure": 1.0}
        
        # Extract value_names using regex to match the words Sure, Impossible, Likely
        value_names = []
        for entry in value_outputs:
            # Find all occurrences of "Sure", "Impossible", or "Likely" using regex
            matches = re.findall(r"\b(Sure|Impossible|Likely)\b", entry, re.IGNORECASE)
            # Normalize the matches to match the keys in value_map
            value_names.extend([match.capitalize() for match in matches])
        
        # If no matches were found, return 0
        if not value_names:
            return 0.0
        
        # Calculate the total value based on the occurrences of "Impossible", "Likely", and "Sure"
        total_score = 0.0
        count = 0
        for name in value_names:
            if name in value_map:
                total_score += value_map[name]
                count += 1
        
        # Return the average score
        if count == 0:
            return 0.0
        average_score = total_score / count
        return average_score
