from src.utils import STaRPipeline
from src.data import QAData

qa_data = QAData()

pipe = STaRPipeline(
    model_id="Ksgk-fy/ecoach_phil_v11_3", 
    roleplay_prompt="Roleplay as Maria, a Filipina woman having a conversation with an FWD insurance agent.",
    qadata=qa_data,
    num_rationales=50
)

print("------- Distilled Reasoning with Human feedback ....")
pipe.process_datapoints()

print("------- Completed. Preparing SFT data ....")
qa_data.prep_data()
print("------- Completed. SFT data prepared.")