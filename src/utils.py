# Get deployed llm to generate responses, then use OAI to correct and supervise the rationale
# --- we focus on the wrong ones 
# Self-Taught Reasoner for association enhancement 
import random
from openai import OpenAI 
from os import getenv 
import json
import lmdeploy

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
    
    def __init__(self, question, answer_choices, correct_answer, pipe, oai_client):
        self.question = question
        self.answer_choices = answer_choices
        self.correct_answer = correct_answer
        self.generated_rationale = ""
        self.generated_answer = ""
        self.oai_client = oai_client
        self.pipe = pipe
        self.correct_rationales = [] # Record of all the correct rationales

    def generate_rationale_and_answer(self, use_lm=False):
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
        is_valid = self.generated_answer and self.generated_rationale
        is_correct = self.generated_answer.lower() == self.correct_answer.lower()
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
    
    def __init__(self, model_id, roleplay_prompt, data, num_rationales=5):
        self.model_id = model_id
        self.pipe = lmdeploy.pipeline(model_id)
        self.oai_client = OpenAI(api_key=API_KEY)
        self.roleplay_prompt = roleplay_prompt
        self.num_rationales = num_rationales

        # Initialize STaR datapoints here
        self.datapoints = [STaRDatapoint(q, ac, a, self.pipe, self.oai_client) 
                           for q, ac, a in zip(data["question"], data["answer_choices"], data["answer"])]

    def process_datapoints(self):
        for datapoint in self.datapoints:
            # 1. Use the to-be-trained model to generate rationale and answer
            response = datapoint.generate_rationale_and_answer(use_lm=True)
            
            # 2. Check if the answer is correct
            if not datapoint.check_answer():
                print("Wrong Answer")
                # If wrong, use strong LLM (OpenAI) to generate multiple rationales
                for _ in range(self.num_rationales):
                    # Prompt the strong LLM (OpenAI) to provide rationale for the correct answer
                    prompt = f"""
                    Question: {datapoint.question}
                    
                    The correct answer is: {datapoint.answer_choices[ord(datapoint.correct_answer) - ord('a')][3:]}

                    Please provide a concise rationale explaining why this is the correct answer. No need to mention the answer.
                    """
                    response = get_oai_response(prompt, system_prompt="You are an expert in reasoning.", oai_client=self.oai_client)
                    
                    # Extract the rationale from the response
                    strong_rationale = response
                    
                    # Update the datapoint with the strong rationale and correct answer
                    datapoint.generated_rationale = strong_rationale
                    datapoint.generated_answer = datapoint.correct_answer
                    
                    # Add the rationale to correct_rationales
                    datapoint.correct_rationales.append(strong_rationale)
                    print("Strong Rationale: ", strong_rationale)
                    
            else:
                # 3. If correct, record the rationale
                datapoint.correct_rationales.append(datapoint.generated_rationale)

