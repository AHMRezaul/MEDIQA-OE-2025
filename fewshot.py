import os
import json
from typing import List, Dict

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
HF_TOKEN = "your_hf_token_here"  # Replace with your Hugging Face token
CACHE_DIR = "../.cache"

TEST_FILE    = "data/test_output_data.json"
FEWSHOT_FILE = "data/train.json"
OUTPUT_FILE  = "result/order_preds.txt"

SYSTEM_MSG = (
    "You are a clinical assistant specialized in extracting medical orders from transcripts.\n"
    "Your job is to identify expected clinical orders from doctor-patient conversations.\n"
    "Return each order on a new line using the following comma-separated format wrapped with '$' symbols:\n"
    "${order_type, description, reason, provenance}$\n"
    "- order_type must be one of [followup, imaging, lab, medication]\n"
    "- description: short, simple description of medical condition from the transcript\n"
    "- reason: concise reason for the description taken exactly from the transcript\n"
    "- provenance: list of the turn_id(s) of the utterance(s) containing the reason, separated by commas for multiple\n"
    "Use the keyword 'null' for any missing field.\n"
    "Only return plain text in the specified format.\n"
    "There might be multiple clinical orders in the conversation.\n"
)

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def format_example(example: Dict) -> str:
    lines = []
    for turn in example["transcript"]:
        speaker = turn["speaker"]
        tid = turn["turn_id"]
        text = turn["transcript"]
        lines.append(f"[{tid}] {speaker}: {text}")
    return "\n".join(lines)

def build_chat(example: Dict, exemplars: List[Dict]) -> List[Dict]:
    def txt(text: str) -> Dict:
        return {"type": "text", "text": text}

    messages: List[Dict] = [{"role": "system", "content": [txt(SYSTEM_MSG)]}]

    for ex in exemplars[1:2]:  # single exemplar, no output labels
        example_text = format_example(ex)
        messages.append({"role": "user", "content": [txt(example_text)]})
        messages.append({"role": "assistant", "content": [txt("...")]})  # no answer given

    current_text = format_example(example)
    messages.append({"role": "user", "content": [txt(current_text)]})
    return messages

def parse_orders(text: str) -> List[Dict]:
    orders = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue  # skip malformed lines

        order_type = parts[0] if parts[0] != "null" else None
        description = parts[1] if parts[1] != "null" else None
        reason = parts[2] if parts[2] != "null" else None

        provenance_raw = parts[3:]
        provenance = []
        for p in provenance_raw:
            try:
                provenance.append(int(p))
            except:
                continue

        orders.append({
            "order_type": order_type,
            "description": description,
            "reason": reason,
            "provenance": provenance
        })
    return orders

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    print("[+] Loading LLaMA-4 model…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=HF_TOKEN)
    model = Llama4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN
    ).eval()

    test_data = load_json(TEST_FILE)
    exemplars = load_json(FEWSHOT_FILE)

    ensure_dir(OUTPUT_FILE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for case in tqdm(test_data, desc="Generating"):
            messages = build_chat(case, exemplars)
            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            try:
                inputs = processor(text=chat_text, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
            except Exception as e:
                print(f"[ERROR] Preprocessing failed for ID {case.get('id', '?')}: {e}")
                continue

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.2,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                output_block = f"\n#{case['id']}#\n{generated_text}\n{'-'*80}\n"
                print(output_block)
                fout.write(output_block)
            except Exception as e:
                output_block = f"\n#{case['id']}#\n$null,null,null,null$\n{'-'*80}\n"
                print(f"[ERROR] Generation failed for ID {case.get('id', '?')}: {e}")
                fout.write(output_block)

if __name__ == "__main__":
    main()
