# MEDIQA-OE: Medical Order Extraction

This repository contains a pipeline for extracting structured medical orders from doctor-patient conversations using large language models.

## 📁 Dataset

The `data/` directory contains the dataset files used for few-shot prompting and evaluation:

- `train.json`: Contains few-shot exemplars extracted from the full dataset.
- `test_output_data.json`: Used for evaluation; contains the `id` and `transcript` fields for each test instance.

## 🚀 Running the Pipeline

1. **Set up the environment**

Create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

2. **Generate predictions**

Run the few-shot inference script with your Hugging Face token by replacing `your_hf_token` with your token:

```bash
python fewshot.py
```

This will generate raw predictions and save them to `result/order_preds.txt`.

3. **Post-process predictions**

Run the post-processing script to clean and format the output:

```bash
python processing.py
```

This will create the final structured predictions in `result/order_preds.json`.

## 🔧 Model

This pipeline uses the [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) model for generation via the Hugging Face Transformers library.

The output was generated using A100 GPUs.

## 📂 Output Format

The final output (`order_preds.json`) contains a mapping from sample `id` to a list of extracted clinical orders. Each order includes:

- `order_type`: One of `[followup, imaging, lab, medication]`
- `description`: Short description of the condition
- `reason`: Exact reasoning from the transcript
- `provenance`: List of `turn_id`s where the reason appears

## 📌 Notes

- Missing or unextractable fields are represented using `null` values.
- If a sample ID in the test set has no predictions, a default entry with `null` fields and empty provenance is inserted.

---

Feel free to open an issue or pull request for suggestions or improvements.
