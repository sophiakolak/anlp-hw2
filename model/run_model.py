import random
from spacy.lang.en import English
import os
from spacy.tokenizer import Tokenizer
import requests

API_TOKEN = env['API_TOKEN']
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

label_map = {"TaskName": "t-n", "DatasetName": "d-n", "HyperparameterValue": "h-v", "HyperparameterName": "h-n",
             "MetricName": "m-n", "MetricValue": "m-v", "MethodName": "mt-n", "MethodValue": "mt-v", "O": 'o'}
reverse_label_map = {v: k for k, v in label_map.items()}
ignore = ["(", ")", ":", " ", "-", "_", ",", "."]

nlp = English()
nlp.tokenizer = Tokenizer(nlp.vocab)

def get_files(path, ext):
    files = []
    for file in os.listdir(path):
        if file.endswith(ext):
            files.append(os.path.join(path, file))
    return files

def format_word(word):
    return f"('{word}',"

def format_label(label):
    if label == 'O':
        return 'o'
    return label.strip()[2:]

def get_ctx(path):
    with open(path, 'r') as f:
        ctx = ""
        words, labels = [], []
        for line in f:
            if line.strip() == '':
                continue
            word, label = line.strip().split(' ')
            if label == 'O':
                label = 'o'
            else:
                label = label.split('-')[1]
                label = label_map[label]
            if word in words:
                continue
            words.append(word)
            ctx += f"('{word}',{label})"
        f.close()
        return ctx

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def get_label(word):
    input = get_ctx('data/train/s1.conll') + format_word(word)

    output = query(
        {
            "inputs": input,
            "parameters": {"max_length": 10, "temperature": 0.1, "do_sample": True},
        }
    )

    generated_text = output[0]["generated_text"]
    response_only = generated_text.split(input)[1].split(')')[0]
    return response_only


def main():
    path = 'data/test'
    files = get_files(path, '.txt')
    name_words = {}
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        if name == 's3':
            continue
        with open(file, 'r') as f:
            data = f.read()
            words = nlp(data)
        words = [word.text for word in words]
        words = list(filter(lambda x: x != '\n', words))
        words = list(filter(lambda x: x.strip() != "", words))
        words = list(filter(lambda x: x != '\t', words))

        name_words[name] = words

    files = get_files(path, '.conll')
    file_data = {}
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        if name == 's3':
            continue
        if name not in file_data:
            file_data[name] = []
        with open(file, 'r') as f:
            data = f.readlines()
        data = list(filter(lambda x: x != '\n', data))
        words = [line.split(' ')[0] for line in data]
        labels = [line.split(' ')[-1].strip() for line in data]

        tokenized_words = name_words[name]

        for word, label in zip(tokenized_words, labels):
            file_data[name].append((word, label))

    for name in file_data:
        count = 0
        for word, gt_label in file_data[name]:
            if count >= 10:
                exit()
            gt_label = format_label(gt_label)
            if gt_label == 'o':
                continue
            label = get_label(word)
            print(f'word={word}, gt_label={gt_label}, label={reverse_label_map[label]}')
            count += 1

if __name__ == '__main__':
    main()