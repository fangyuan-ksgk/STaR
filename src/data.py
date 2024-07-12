import os
import json 

class InsuranceQAData:
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.file_name = "insurance_qa.json"
        self.data = {
            "question": [],
            "answer_choices": [],
            "answer": [],
            "rationale": []
        }

    def add_qa(self, question, answer_choices, correct_answer, rationale):
        self.data["question"].append(question)
        self.data["answer_choices"].append(answer_choices)
        self.data["answer"].append(correct_answer)
        self.data["rationale"].append(rationale)

    def store(self):
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, self.file_name)
        with open(file_path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved to {file_path}")

    def load(self):
        file_path = os.path.join(self.data_dir, self.file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.data = json.load(f)
            print(f"Data loaded from {file_path}")
        else:
            print(f"File not found: {file_path}")

    def get_data(self):
        return self.data