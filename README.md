Usage:
- Make sure you are running on an environment with GPUs.
- Run `accelerate config` to configure your accelerate settings (optional)
- Run `pip install -r requirements.txt`
- Run `accelerate launch {pipelines}.py` or `python {pipelines}.py`

pipelines includes:
- gpt_pipeline
- llama2_pipeline
- roberta_hotpot_pipeline
