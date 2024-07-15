import os
import json 

class QAData:
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.file_name = "qa.json"
        # Original data
        self.data = {
            "question": [],
            "answer_choices": [],
            "answer": [],
            "rationale": []
        }
        self.star_file_name = "star_qa.json"
        # Augmented data with STaR
        self.star_data = {
            "question": [],
            "answer_choices": [],
            "answer": [],
            "rationale": []
        }
        self.load()

    def add_qa(self, question, answer_choices, correct_answer, rationale, to_original=True):
        """ 
        Adding to the original data / star data
        """
        if to_original:
            self.data["question"].append(question)
            self.data["answer_choices"].append(answer_choices)
            self.data["answer"].append(correct_answer)
            self.data["rationale"].append(rationale)
        else:
            self.star_data["question"].append(question)
            self.star_data["answer_choices"].append(answer_choices)
            self.star_data["answer"].append(correct_answer)
            self.star_data["rationale"].append(rationale)

    def store(self):
        """ 
        TBD: do not reload augmented data, only write extra data to it
        """
        # store original data
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, self.file_name)
        with open(file_path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved to {file_path}")
        
        # store augmented data
        file_path = os.path.join(self.data_dir, self.star_file_name)
        with open(file_path, "w") as f:
            json.dump(self.star_data, f, indent=2)
        print(f"Data saved to {file_path}")

    def load(self):
        # load original file
        file_path = os.path.join(self.data_dir, self.file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.data = json.load(f)
            print(f"Data loaded from {file_path}")
        else:
            print(f"File not found: {file_path}")
            
        # load augmented file
        file_path = os.path.join(self.data_dir, self.star_file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.star_data = json.load(f)
            print(f"Data loaded from {file_path}")

    def get_data(self):
        return self.data
    
    def prep_data(self):
        """ 
        Prepare data for training
        - These data focues on questions which SLM picks the wrong answer (targeted)
        - Augmented data will be used for training 
        - We formulate this into a single-choice classification task, requiring rationale and answer
        """
        messages = []
        for i in range(len(self.star_data["question"])):
            msg = []
            
            question = self.star_data["question"][i]
            answer_choices = self.star_data["answer_choices"][i]
            correct_answer = self.star_data["answer"][i]
            rationale = self.star_data["rationale"][i]
            
            # This is the desired prompt 
            roleplay_prompt = "Roleplay as Maria, a Filipina woman having a conversation with an FWD insurance agent."
            system_prompt = f"{roleplay_prompt} You are Maria, responding to queries from an FWD insurance agent."
            
            if hasattr(self, 'hint_rationale'):
                system_prompt += f" Hint: {self.hint_rationale}"
                
            user_prompt = f"FWD agent asks: {question}\nPossible responses: {answer_choices}\nAs Maria, provide your rationale and choose an answer. Your response should be in the format:\nRationale: [Your rationale here]\nAnswer: [Single letter a/b/c/d corresponding to your chosen response]"
            
            # Formulate the expected response
            correct_answer_index = next(i for i, choice in enumerate(answer_choices) if choice.startswith(f"{correct_answer}."))
            response = f"Rationale: {rationale}\nAnswer: {chr(97 + correct_answer_index)}"
            # Add to messages
            msg.append({"role": "system", "content": system_prompt})
            msg.append({"role": "user", "content": user_prompt})
            msg.append({"role": "assistant", "content": response})
            messages.append(msg)
        
        # Save messages list 
        with open("data/msg_list.json", "w") as f:
            json.dump(messages, f)
        
        return messages