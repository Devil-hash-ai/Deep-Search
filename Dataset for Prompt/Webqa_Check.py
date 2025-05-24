import json

webqa_path = "/home/featurize/WEBQA/WebQA_train_val.json"
prompt_path = "/home/featurize/WEBQA/grounded_sam_prompts/prompts_fallback.jsonl"

with open(webqa_path, "r", encoding="utf-8") as f:
    webqa_data = json.load(f)
print(" WebQA entries:", len(webqa_data))

with open(prompt_path, "r", encoding="utf-8") as f:
    prompts = f.readlines()
print(" Prompt lines:", len(prompts))