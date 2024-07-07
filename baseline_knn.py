import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
#model = AutoModel.from_pretrained('microsoft/deberta-v3-large')

def load_model_and_tokenizer(model_name='microsoft/deberta-v3-large'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def extract_features(text, tokenizer, model):
    # Tokenize the input text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
        model.cuda()
    
    # Get embeddings (output of the last hidden layer)
    with torch.no_grad():  # Disable gradient calculation to save memory and speed up
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # Get the mean of the embeddings for the input tokens to represent the whole input
    mean_embeddings = last_hidden_states.mean(dim=1).squeeze()
    return mean_embeddings  # Move back to CPU and convert to numpy array for general use

tokenizer, model = load_model_and_tokenizer()

data = torch.load('dataset.pt')

import random
random.seed(1234)


import pytz
from datetime import datetime

def convert_to_weekday(gmt_time_str):
    # Parse the timestamp string into a datetime object
    gmt_time = datetime.strptime(gmt_time_str, '%Y-%m-%dT%H:%M:%SZ')
    
    # Define the GMT and Eastern timezones
    gmt_tz = pytz.timezone('GMT')
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Localize the GMT time and convert to Eastern Time
    gmt_time = gmt_tz.localize(gmt_time)
    eastern_time = gmt_time.astimezone(eastern_tz)
    
    # Return the day of the week
    return eastern_time.strftime('%A')  # %A returns the full weekday name


def convert_to_features(x):
    title = x['title']
    authors = x['authors']
    authors = ', '.join(authors)
    abstract = x['abstract']
    published = x['published']
    day = convert_to_weekday(published)
    s = f"""Published on: {day}
Title: {title}
Authors: {authors}
Abstract: {abstract}"""
    features = extract_features(s, tokenizer, model)
    return features


# Get training features
train_labels = []
train_features = []
import tqdm
for x, y in tqdm.tqdm(data['train']):
    features = convert_to_features(x)
    train_features.append(features)
    train_labels.append(1 if y else 0)
    #if len(train_labels) > 100:
    #    break

#import pdb; pdb.set_trace()
train_features = torch.stack(train_features, dim=0)
train_features = train_features / train_features.norm(dim=-1, keepdim=True)
train_labels = torch.LongTensor(train_labels).view(-1).to(train_features.device)
torch.save({'train_features': train_features, 'train_labels': train_labels}, 'baseline_knn.pt')
precisions = []
recalls = []
repeats = 1
ks = list(range(1, 32, 2))          # k values from 1 to 20
for k in ks:
    tp = 0
    data_positive = 0
    pred_positive = 0
    for _ in range(repeats):
        for x, y in tqdm.tqdm(data['test']):
            f = convert_to_features(x)
            scores = train_features @ f
            top_values, top_ids = torch.topk(scores, k)
            top_labels = train_labels.gather(0, top_ids)
            counter = Counter(top_labels.cpu().tolist())
            pred_label, _ = counter.most_common(1)[0]
            if pred_label > 0:
                predicted = True
            else:
                predicted = False
            if predicted and y:
                tp += 1
            if y:
                data_positive += 1
            if predicted:
                pred_positive += 1
            #if data_positive > 3:
            #    break
    precision = tp / max(1e-6, pred_positive)
    recall = tp / max(1e-6, data_positive)
    precisions.append(precision)
    recalls.append(recall)

#print (ks)
#print (precisions)
#print (recalls)
#import matplotlib.pyplot as plt
#plt.plot(precisions, recalls, '.')
#plt.xlabel('precision')
#plt.ylabel('recall')
#plt.savefig('baseline_knn.png')


# Create a color map and normalize
norm = plt.Normalize(min(ks), max(ks))  # Normalization for the color map
colors = cm.viridis(norm(ks))  # Get color map colors based on normalized k values

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(precisions, recalls, color=colors, s=50)  # Plot scatter with color

# Adding annotations for each point
for i, k in enumerate(ks):
    ax.text(precisions[i], recalls[i], f'{k}', fontsize=9, ha='right', va='center')

# Adding labels and title
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
#ax.set_title('Precision vs Recall for Different k in k-NN')

# Create a ScalarMappable and Colorbar
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])  # You need to set_array([]) to avoid warnings in some Matplotlib versions
cbar = plt.colorbar(sm, ax=ax, aspect=10)  # Link the colorbar to the ax
cbar.set_label('Value of k')

plt.savefig('baseline_knn.png')
