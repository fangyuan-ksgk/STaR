from src.utils import STaRPipeline
from src.data import InsuranceQAData

qa_data = InsuranceQAData()

pipe = STaRPipeline(
    model_id="Ksgk-fy/ecoach_phil_v11_3", 
    roleplay_prompt="Roleplay as Maria, a Filipina woman having a conversation with an FWD insurance agent.",
    data=qa_data.data,
    num_rationales=50
)

pipe.process_datapoints()