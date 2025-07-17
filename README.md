# mediqa-oe
Medical Order Extraction

## Dataset
The data folder contains the datasets. The ```train.json``` file is extracted from the dataset files and used for fewshot example.
The ```test_output_data.json``` file is used for the testing purposes. It contains all the test data ```id``` and ```transcript```.

## Running
Install all the required libraries in a virtual environment from the requirements.txt file.

Run the ```fewshot.py``` script with ```your_hf_token``` (Your huggingface token). 

The output is saved in the ```result``` folder as a ```.txt``` file.

Next, run the ```processing.py``` script. This will generate the final ```order_preds.json``` file inside the ```result``` folder.

## Tools
Used ```meta-llama/Llama-4-Scout-17B-16E-Instruct``` as the generation model. 

