import pytz
from datetime import datetime
from dataclasses import dataclass
import os
import copy
import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
random.seed(1234)

def convert_to_weekday(gmt_time_str):
    gmt_time = datetime.strptime(gmt_time_str, '%Y-%m-%dT%H:%M:%SZ')
    gmt_tz = pytz.timezone('GMT')
    eastern_tz = pytz.timezone('US/Eastern')
    gmt_time = gmt_tz.localize(gmt_time)
    eastern_time = gmt_time.astimezone(eastern_tz)
    return eastern_time.strftime('%A')  # %A returns the full weekday name

def convert_to_features(x):
    title = x['title'].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').strip()
    authors = x['authors']
    authors = ', '.join(authors)
    abstract = x['abstract'].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').strip()
    published = x['published']
    day = convert_to_weekday(published)
    time = published.split('T')[-1]
    categories = x['categories']
    categories = '; '.join(categories)
    s = f"""Published: {day} {time}
Title: {title}
Authors: {authors}
Abstract: {abstract}"""
    return s


def upsample_to_target_ratio(data, target_ratio=0.3):
    # Separate the majority and minority classes
    class_0 = [(text, label) for text, label in data if label == 0]
    class_1 = [(text, label) for text, label in data if label == 1]
    
    # Check which one is minority and majority
    minority, majority = (class_1, class_0) if len(class_1) < len(class_0) else (class_0, class_1)
    minority_label = 1 if len(class_1) < len(class_0) else 0

    # Calculate target minority count
    majority_count = len(majority)
    desired_minority_count = int(target_ratio * majority_count / (1 - target_ratio))

    # Calculate how many samples to add
    samples_to_add = desired_minority_count - len(minority)
    
    if samples_to_add > 0:
        # Randomly sample from the minority class with replacement
        additional_samples = random.choices(minority, k=samples_to_add)
        multiplier = desired_minority_count / len(minority)
        balanced_dataset = majority + minority + additional_samples
    else:
        balanced_dataset = majority + minority
        multiplier = 1

    # Shuffle the combined dataset to mix the upsampled samples
    random.shuffle(balanced_dataset)
    return balanced_dataset, multiplier


class AZDataset(Dataset):
    def __init__(self, tokenizer, data_split, max_length, target_ratio=None):
        print ('Creating Features')
        inputs_all = []
        self.multiplier = 1
        if target_ratio is not None:
            #import pdb; pdb.set_trace()
            print(f'Upsampling to {target_ratio}')
            data_split, multiplier = upsample_to_target_ratio(data_split, target_ratio)
            self.multiplier = multiplier
        #import pdb; pdb.set_trace()
        for x, y in tqdm.tqdm(data_split):
            text = convert_to_features(x)
            text = f'[CLS] {text} [SEP]'
            inputs = tokenizer([text], add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
            inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': torch.LongTensor([y]).view(1)}
            inputs_all.append(inputs)
        self.data = inputs_all
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return self.data[i]

@dataclass
class AZDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = {}
        for k in examples[0]:
            batch[k] = []
        for example in examples:
            for k in batch:
                batch[k].append(example[k])
        for k in batch:
            batch[k] = torch.cat(batch[k], dim=0)
        return batch
