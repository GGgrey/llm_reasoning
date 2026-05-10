import re
import os

import sympy
import pandas as pd
from fuzzywuzzy import fuzz

from tasks.base import Task, DATA_PATH
from prompts.bamboogle import * 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Bamboogle(Task):
    def __init__(self, file='Bamboogle Prerelease - Sheet1.csv'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'bamboogle', file)
        if not os.path.isfile(path):
            raise ValueError(f"File {file} not found at {path}")
        try:
            self.data = list(pd.read_csv(path, encoding='utf-8')['Question'])
            self.ground_truth = list(pd.read_csv(path, encoding='utf-8')['Answer'])
        except UnicodeDecodeError:
            self.data = list(pd.read_csv(path, encoding='ISO-8859-1')['Question'])
            self.ground_truth = list(pd.read_csv(path, encoding='ISO-8859-1')['Answer'])

        self.value_cache = {}
        self.steps = 3
        self.stops = ['.', '.', 'End of answer.']

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

        print(f'==== Ground Truth: {ground_truth} ====')
        print(f'==== Extracted Answer: {expression} ====')
        
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
        
        if output.lower().startswith('question:'):
            second_question_index = output.lower().find('question:', len('question:'))
            if second_question_index != -1:
                output = output[:second_question_index].strip()
        else:
            question_index = output.lower().find('question:')
            if question_index > -1:
                output = output[:question_index].strip()

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
            return wiki_evaluate + x + '\nThought Process: ' + y + '\nEvaluation Process:\n'
        else:
            if 'choose the best answer' in x.lower():
                return choose_evaluate + x + '\nAnswer: ' + y.lower().split('the final answer is')[1].replace(': ', '') + '\nEvaluation Process:\n'
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

        print("\n===== Value Outputs =====")
        for i, value_output in enumerate(value_outputs):
            print(f"Output {i + 1}: {value_output}")
        print("=========================\n")
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