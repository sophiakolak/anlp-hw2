# CMU Advanced NLP Assignment 2: End-to-end NLP System Building

by Sophia Kolak, Daniel Ramos

## Methodology 

For this assignment, we used few-shot prompting (rather than fine-tuning) with [bloom](https://huggingface.co/bigscience/bloom) to obtain our test results. As such, there was no training process. The classifaction task instead involves pre-processing, building the prompt, querying, and post-processing. For testing, we used manually annotated data from scraped ACL papers. For prompt engineering, we used the annoted data provided in the assignment repository.  

## Running 

Steps required to reproduce results:

``` git clone https://github.com/sophiakolak/anlp-hw2 ```

``` cd anlp-hw2 ```

``` python -m venv <ENV_NAME> ```

``` source <ENV_NAME>/bin/activate ```

``` pip3 install requirements.txt ```

``` python3 model/run_model.py ```

## `model` 

`model` contains the code to run the query on the LLM (bloom) used for this assignment. 

## `scrape`

`scrape` contains files used to mine the ACL Anthology, where we obtained papers to annotate. 
