import os
import json 

class InsuranceQAData:
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.file_name = "insurance_qa.json"
        # Original data
        self.data = {
            "question": [],
            "answer_choices": [],
            "answer": [],
            "rationale": []
        }
        self.star_file_name = "star_insurance_qa.json"
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