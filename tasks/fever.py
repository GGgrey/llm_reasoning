import re
import os
import json

import pandas as pd
from fuzzywuzzy import fuzz

from tasks.base import Task, DATA_PATH
from prompts.fever import * 


wiki_evaluate = '''Evaluate whether the language model can effectively decompose the claim into relevant sub-questions, and assess whether this decomposition helps in partially or directly verifying the original claim. The outcome will determine if this process of decomposition is "Likely" or "Impossible" to aid in verifing the claim.

Evaluation Steps: Check if the language model can identify and decompose key sub-questions that are directly related to the original question.
Evaluation Process: 1. Analyze whether each sub-question identified by the model is directly relevant to the verify the original claim. 2. Determine if the decomposition of these sub-questions forms a reasonable verification process to the original claim.
Evaluation Result: 1. Likely: If the language model successfully decomposes the original claim into relevant sub-questions that help construct the final answer. 2. Impossible: If the language model fails to effectively decompose the claim, or if the decomposed sub-questions are not directly relevant to make the verification.

Claim: Reg Watson is a current television producer.
Thought Process: Step 1, who is Reg Watson? Reginald James Watson AM was an Australian television producer and screenwriter.
Evaluation Process:
Relevance of Sub-Questions: The sub-question of identifying who Reg Watson is, is directly relevant as it establishes the necessary context to further explore the original claim about his current status as a television producer.
Effectiveness of Decomposition: By first identifying Reg Watson and then investigating his current professional activities, this approach forms a reasonable verification process. 
Evaluation Result:
Likely

Claim: The Gadsden flag was named by Christopher Gadsden.
Thought: Step 1, why did Christopher Gadsden die? Gadsden died from head injuries suffered in a fall near his home.
Evaluation Process:
Relevance of Sub-Questions: The sub-question about the cause of Gadsden's death does not directly contribute to verifying the claim about the naming of the Gadsden flag. Instead, questions should focus on his involvement with the flag and any direct actions or contributions he made towards its naming.
Verification Process: A more effective verification process would involve gathering evidence of Christopher Gadsden's direct involvement with the flag, including any documented instances where he is credited with its naming, as well as understanding the historical context of the flag's creation and use.
Evaluation Results:
Impossible

Claim: Black Mirror is about society.
Thought Process: Step 1, what is the son of Black Mirror? Black Mirror is a British anthology television series. Step 2, what issues does this series discuss? The series uses technology to comment on contemporary social issues.
Evaluation Process:
Relevance of Sub-Questions: Each sub-question is directly relevant and helps verify the original claim. The first establishes the series' nature and scope, and the second addresses the thematic content, specifically its societal focus.
Verification Process: This process is reasonable for verifying the original claim. First, it establishes what "Black Mirror" is, laying the groundwork for further inquiry. Then, it dives into the series' thematic concerns, confirming its focus on societal issues through the lens of technology. This approach not only verifies the claim but also provides insight into how the series approaches its critique of society.
Evaluation Results:
Likely

Claim: '''

Final_evaluate = '''Evaluate whether the conclusion can be drawn based on reasoning logic in the thought process." (Likely/Impossible).

Claim: The Gadsden flag was named by Christopher Gadsden.
Thought Process: Step 1, why did Christopher Gadsden die? Gadsden died from head injuries suffered in a fall near his home. Step 2, what is the origin of the name of the Gadsden flag? The Gadsden flag is named after politician Christopher Gadsden. 
Conclusion: so the final answer is: REFUTES.
Evaluation Process:
The thought process includes a correct statement about the origin of the Gadsden flag's name that aligns with the claim, but then concludes incorrectly that this information refutes the claim.
Evaluation Results:
Impossible

Claim: Black Mirror is about society.
Thought Process: Step 1, what is the son of Black Mirror? Black Mirror is a British anthology television series. Step 2, what issues does this series discuss? The series uses technology to comment on contemporary social issues.
Conclusion: so the final answer is: SUPPORTS.
Evaluation Process:
The conclusion logically follows from the information provided, supporting the claim that "Black Mirror" is about society.
Evaluation Results:
Likely

Claim: '''


class FactualQA(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='paper_test.jsonl'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'fever', file)
        self.data = []
        self.ground_truth = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                question = item.get("claim")
                answer = item.get("label")

                self.data.append(question)
                self.ground_truth.append(answer)
        
        self.value_cache = {}
        self.steps = 3
        self.stops = ['.', '.','Question'] 

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
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
