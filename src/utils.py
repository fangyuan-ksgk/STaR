# Get deployed llm to generate responses, then use OAI to correct and supervise the rationale
# --- we focus on the wrong ones 
# Self-Taught Reasoner for association enhancement 
import random
from openai import OpenAI 
from os import getenv 
import json
import lmdeploy
import os
from .data import InsuranceQAData

# Model & LLM 
# LMDeploy + OpenAI (Student & Teacher Pair)
# model_id = "Ksgk-fy/ecoach_phil_v11_3"
# pipe = lmdeploy.pipeline(model_id)

API_KEY = getenv("OPENAI_API_KEY")

def get_oai_response(prompt, system_prompt, oai_client):
    response = oai_client.chat.completions.create(
        model="gpt-4o",  # Use an appropriate OpenAI model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content


def get_lmdeploy_response(prompts, pipe):
    """ 
    Batch inference with LMDeploy package, local LLM
    """
    import asyncio
    import nest_asyncio
    
    async def run_inference():
        responses = pipe(prompts)
        return [response.text for response in responses]
    
    # Apply nest_asyncio to allow running event loop in Jupyter
    nest_asyncio.apply()
    
    return asyncio.run(run_inference())


def get_vllm_response(prompts, llm):
    """ 
    Batch Inference with vLLM, local LLM
    """
    import asyncio
    import nest_asyncio
    
    async def run_inference():
        responses = llm.generate(prompts)
        return [response.text for response in responses]
    
    # Apply nest_asyncio to allow running event loop in Jupyter
    nest_asyncio.apply()
    
    return asyncio.run(run_inference())
    

# Util function
def parse_rationale_answer(response):
    """
    Parse out Rationale and Answer from the response text.
    Returns False, False if parsing fails.
    """
    rationale, answer = False, False
    try:
        ration_suffix = response.split("Rationale: ")[1]
        rationale = ration_suffix.split("Answer: ")[0].strip()
        answer_txt = ration_suffix.split("Answer: ")[1].split("\n")[0].strip()
        if "a." in answer_txt.lower() or "a" == answer_txt:
            answer = "a"
        elif "b." in answer_txt.lower() or "b" == answer_txt:
            answer = "b"
        elif "c." in answer_txt.lower() or "c" == answer_txt:
            answer = "c"
        elif "d." in answer_txt.lower() or "d" == answer_txt:
            answer = "d"        
    except:
        pass
    return rationale, answer


roleplay_prompt = "Roleplay as Maria, a Filipina woman having a conversation with an FWD insurance agent."


class STaRDatapoint:
    
    def __init__(self, question, answer_choices, correct_answer, rationale, pipe, oai_client):
        self.question = question
        self.answer_choices = answer_choices
        self.correct_answer = correct_answer
        self.hint_rationale = rationale
        self.generated_rationale = ""
        self.generated_answer = ""
        self.oai_client = oai_client
        self.pipe = pipe
        self.correct_rationales = [] # Record of all the correct rationales

    def generate_rationale_and_answer(self, use_lm=False, use_hint=False):
        if use_hint:
            system_prompt = f"{roleplay_prompt} You are Maria, responding to queries from an FWD insurance agent. Hint: {self.hint_rationale}"
        else:
            system_prompt = f"{roleplay_prompt} You are Maria, responding to queries from an FWD insurance agent."
            
        user_prompt = f"FWD agent asks: {self.question}\nPossible responses: {self.answer_choices}\nAs Maria, provide your rationale and choose an answer. Your response should be in the format:\nRationale: [Your rationale here]\nAnswer: [Single letter a/b/c/d corresponding to your chosen response]"
        
        if use_lm:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = get_lmdeploy_response([full_prompt], self.pipe)[0]
        else:
            response = get_oai_response(user_prompt, system_prompt, self.oai_client)

        rationale, answer = parse_rationale_answer(response)
        self.generated_rationale = rationale
        self.generated_answer = answer 
        print("Generated Answer: ", answer)
        return response

    def check_answer(self):
        """ 
        Check if the generated answer is correct with valid rationale
        """
        is_valid = self.generated_answer and self.generated_rationale
        try:
            is_correct = self.generated_answer.lower() == self.correct_answer.lower()
        except:
            is_correct = False
        return is_valid and is_correct 
    
    def generate(self, n, use_lm=False):
        """ 
        Generate n (rationale, answer) tuples
        this will be used to supervise on a LLM to be able to learn the rationale and picking the correct answer
        """
        generated_data = []
        for _ in range(n):
            # Generate a random question and answer choices
            question = f"Random question {_+1}"
            answer_choices = [f"Option {chr(97+i)}" for i in range(4)]  # a, b, c, d
            correct_answer = random.choice(['a', 'b', 'c', 'd'])
            
            # Generate rationale and answer
            self.question = question
            self.answer_choices = answer_choices
            self.correct_answer = correct_answer
            self.generate_rationale_and_answer(use_lm)
            
            # Append to the list
            generated_data.append((self.generated_rationale, self.generated_answer))
        
        return generated_data
    
    
class STaRPipeline:
    
    def __init__(self, 
                 model_id: str, 
                 roleplay_prompt: str, 
                 qadata: InsuranceQAData, 
                 num_rationales: int = 5):
        
        self.model_id = model_id
        self.pipe = lmdeploy.pipeline(model_id)
        self.oai_client = OpenAI(api_key=API_KEY)
        self.roleplay_prompt = roleplay_prompt
        self.num_rationales = num_rationales
        self.qadata = qadata


        # Initialize STaR datapoints here
        data = self.qadata.data
        self.datapoints = [STaRDatapoint(q, ac, a, r, self.pipe, self.oai_client) 
                           for q, ac, a, r in zip(data["question"], data["answer_choices"], data["answer"], data["rationale"])]

    def process_datapoints(self):
        for datapoint in self.datapoints:
            # 1. Use the to-be-trained model to generate rationale and answer
            response = datapoint.generate_rationale_and_answer(use_lm=True)
            
            # 2. Check if the answer is correct
            if not datapoint.check_answer():
                print("Wrong Answer")
                # If wrong, use strong LLM (OpenAI) to generate multiple rationales
                for _ in range(self.num_rationales):
                    response = datapoint.generate_rationale_and_answer(use_lm=False, use_hint=True)
                          
                    # Generate rationale using strong LLM
                    strong_rationale = datapoint.generated_rationale
                    
                    # Check if the answer is correct
                    if datapoint.check_answer():
                        # If correct, update the datapoint and break the loop
                        datapoint.generated_rationale = strong_rationale
                        datapoint.correct_rationales.append(strong_rationale)
                        print("Strong Rationale: ", strong_rationale)
                        self.qadata = self.save(datapoint, self.qadata)
                    
                    # If incorrect, continue to next iteration to generate another rationale
                    
            else:
                # 3. If correct, record the rationale
                datapoint.correct_rationales.append(datapoint.generated_rationale)
                self.qadata = self.save(datapoint, self.qadata)
                
        # Update the qadata into local database
        self.qadata.save()
            
    def save(self, datapoint: STaRDatapoint, qadata: InsuranceQAData):
        """ 
        Save the augmented data to the original data
        """
        # We only want to save the current datapoint, not iterate over all datapoints
        qadata.add_qa(datapoint.question, datapoint.answer_choices, datapoint.correct_answer, datapoint.generated_rationale)
        return qadata 