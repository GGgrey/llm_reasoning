import re
import os
import sympy
import pandas as pd
import json
from tasks.base import Task, DATA_PATH
from prompts.prontoqa import * 
from fuzzywuzzy import fuzz


class ProntoQA(Task):
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
    def __init__(self, file='prontoqa.json'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'ProntoQA', file)
        if not os.path.isfile(path):
            raise ValueError(f"File {file} not found at {path}")
        
        self.data = []
        self.ground_truth = []

        # Load data from JSON file
        with open(path) as f:
            json_data = json.load(f)
            for item in json_data:
                question = item.get("question")
                context = item.get("context")
                final_question = context + " " + question
                self.data.append(final_question)
                answer = item.get("answer")
                answer = "true" if answer == "A" else "false"
                self.ground_truth.append(answer)
        
        self.value_cache = {}
        self.steps = 3
        self.stops = ['.', '.', 'Question']

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for data of length {len(self.data)}")
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        if 'answer is' not in output.lower():
            print('====output====')
            print(output)
            return {'r': 0}
        
        expression = self.extract_answer(output)
        expression = expression.replace(': ', '').strip()
        ground_truth = str(self.ground_truth[idx])

        print(f'====GR===={ground_truth}====Pre===={expression}')
        
        if ground_truth in expression:
            return {'r': 1}
        else:
            expression_ = re.sub(r'\W+', '', expression, flags=re.IGNORECASE)
            ground_truth_ = re.sub(r'\W+', '', ground_truth, flags=re.IGNORECASE)
            
            if re.search(ground_truth_, expression_, re.IGNORECASE):
                return {'r': 1}
            else:
                similarity = fuzz.ratio(expression_, ground_truth_)
                if similarity > 95:
                    return {'r': 1}
                
                ground_truth_parts = ground_truth.split(' ')
                flag = any(re.search(part, expression, re.IGNORECASE) for part in ground_truth_parts)
                return {'r': 1} if flag else {'r': 0}

    def extract_answer(self, output: str) -> str:
        output = output.replace('\n', ' ').strip()
        
        if 'the final answer is' in output.lower():
            return output.lower().split('the final answer is')[-1].strip().split('.')[0].strip()
        elif 'final answer' in output.lower():
            return output.lower().split('final answer')[-1].strip().split('.')[0].strip()
        else:
            return output.strip().split('.')[-1].strip()

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def reflect_cot_prompt_wrap(x: str, y: str = '') -> str:
        return reflect_cot_prompt.format(input=x) + y

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        if 'the final answer is' not in y.lower():
            return value_evaluate + x + '\nThought Process: ' + y + '\nEvaluation Process:\n'
        else:
            return final_evaluate + x + '\n' + y + '\nEvaluation Process: \n'
        
    @staticmethod
    def self_process_value_prompt_wrap(x: str, y: str) -> str:
        return value_evaluate + x + '\nThought Process: ' + y + '\nEvaluation Process:\n'

    @staticmethod
    def self_result_value_prompt_wrap(x: str, y: str) -> str:
        return final_evaluate + x + '\nSo the final answer is:' + y + '\nEvaluation Process: \n'  
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # Value map for scoring with probabilities between 0 and 1
        value_map = {'Impossible': 0.0, 'Likely': 0.5, 'Sure': 1.0}
        
        # Extract value_names using regex to match the words Sure, Impossible, Likely
        value_names = []
        for entry in value_outputs:
            # Find all occurrences of 'Sure', 'Impossible', or 'Likely' using regex
            matches = re.findall(r'\b(Sure|Impossible|Likely)\b', entry, re.IGNORECASE)
            # Normalize the matches to match the keys in value_map
            value_names.extend([match.capitalize() for match in matches])
        
        # If no matches were found, return 0
        if not value_names:
            return 0.0
        
        # Calculate the total value based on the occurrences of 'Impossible', 'Likely', and 'Sure'
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