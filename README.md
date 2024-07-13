# STaR
Unofficial Implementation of Self-Taught Reasoner (STaR)

Present the desired behavior to LLM in the form of sing-choice question answering, adopts STaR-like approach to enhance association. 
Rationalization does not work well, included human rationale as hint to bootstrap its reasoning helps.
SLM is confused from time to time, use strong LLM to get around it.

```shell
python run.py
```

Resulting SFT ready dataset 
```shell
data/msg_list.json
```

Adding to QA dataset 
```python
from src import QAdata
qadata = QAdata()
qadata.add_qa(data)
```


