import re
import json
import os

# === File paths ===
RAW_INPUT_FILE = "result/order_preds.txt"
CLEANED_OUTPUT_FILE = "result/order_preds_cleaned.txt"
FINAL_JSON_FILE = "result/order_preds.json"

# === Pre-processing ===
def run_preprocessing():
    print("[+] Running pre-processing...")
    system_prompt = (
        "system\n\n"
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
        "There might be multiple clinical orders in the conversation.user\n"
    )

    with open(RAW_INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(system_prompt, "")
    content = content.replace("...user\n\n", "")
    content = content.replace("--------------------------------------------------------------------------------", "")

    cleaned_lines = [line for line in content.splitlines() if not line.strip().startswith("[")]

    with open(CLEANED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    print(f"[+] Saved cleaned output to {CLEANED_OUTPUT_FILE}")

# === Post-processing ===
def extract_prediction_blocks(text):
    blocks = re.split(r"#(.*?)#", text)
    output = {}

    for i in range(1, len(blocks) - 1, 2):
        sample_id = blocks[i].strip()
        content = blocks[i + 1]

        predictions = []

        matches = re.findall(r"\$(.*?)\$", content, flags=re.DOTALL)
        for match in matches:
            line = match.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            order_type = parts[0] if parts[0] != "null" else None
            description = parts[1] if parts[1] != "null" else None
            reason = parts[2] if parts[2] != "null" else None

            provenance = []
            for p in parts[3:]:
                try:
                    provenance.append(int(p))
                except:
                    continue

            predictions.append({
                "order_type": order_type,
                "description": description,
                "reason": reason,
                "provenance": provenance
            })

        output[sample_id] = predictions

    return output

def run_postprocessing():
    print("[+] Running post-processing...")
    with open(CLEANED_OUTPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    extracted = extract_prediction_blocks(raw_text)

    with open(FINAL_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=4, ensure_ascii=False)

    print(f"[+] Saved predictions to {FINAL_JSON_FILE}")

# === Main ===
if __name__ == "__main__":
    run_preprocessing()
    run_postprocessing()
