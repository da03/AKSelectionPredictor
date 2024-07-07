import torch
import pytz
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tqdm
from notimedata import convert_to_features
from utils import compute_metrics
import random
random.seed(1234)


#tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
#model = AutoModel.from_pretrained('microsoft/deberta-v3-large')

def load_model_and_tokenizer(model_name='microsoft/deberta-v3-large', tokenizer_name='microsoft/deberta-v3-large'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@torch.no_grad()
def get_logits(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
        model.cuda()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits

tokenizer, model = load_model_and_tokenizer(model_name='model_finetune_deberta_lr2e5_nomultiplier/checkpoint_0_step_1107/')
model.eval()
data = torch.load('dataset.pt')

x_test = data['train'][0][0]
import pdb; pdb.set_trace()
x_test['title'] = 'WildChat: 1M ChatGPT Interaction Logs in the Wild'
x_test['authors'] = 'Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng'.split(', ')
x_test['abstract'] = 'Chatbots such as GPT-4 and ChatGPT are now serving millions of users. Despite their widespread use, there remains a lack of public datasets showcasing how these tools are used by a population of users in practice. To bridge this gap, we offered free access to ChatGPT for online users in exchange for their affirmative, consensual opt-in to anonymously collect their chat transcripts and request headers. From this, we compiled WildChat, a corpus of 1 million user-ChatGPT conversations, which consists of over 2.5 million interaction turns. We compare WildChat with other popular user-chatbot interaction datasets, and find that our dataset offers the most diverse user prompts, contains the largest number of languages, and presents the richest variety of potentially toxic use-cases for researchers to study. In addition to timestamped chat transcripts, we enrich the dataset with demographic data, including state, country, and hashed IP addresses, alongside request headers. This augmentation allows for more detailed analysis of user behaviors across different geographical regions and temporal dimensions. Finally, because it captures a broad range of use cases, we demonstrate the dataset’s potential utility in fine-tuning instruction-following models. WildChat is released at https://wildchat.allen.ai under AI2 ImpACT Licenses.'
#x_test['published'] = '2024-04-26T03:36:00Z'
#x_test['title'] = 'BLINK: Multimodal Large Language Models Can See but Not Perceive'
#x_test['title'] = 'Sample Design Engineering: An Empirical Study of What Makes Good Downstream Fine-Tuning Samples for LLMs'
#x_test['title'] = 'Implicit Chain of Thought Reasoning via Knowledge Distillation'
#x_test['abstract'] = """To augment language models with the ability to reason, researchers usually prompt or finetune them to produce chain of thought reasoning steps before producing the final answer. However, although people use natural language to reason effectively, it may be that LMs could reason more effectively with some intermediate computation that is not in natural language. In this work, we explore an alternative reasoning approach: instead of explicitly producing the chain of thought reasoning steps, we use the language model's internal hidden states to perform implicit reasoning. The implicit reasoning steps are distilled from a teacher model trained on explicit chain-of-thought reasoning, and instead of doing reasoning "horizontally" by producing intermediate words one-by-one, we distill it such that the reasoning happens "vertically" among the hidden states in different layers. We conduct experiments on a multi-digit multiplication task and a grade school math problem dataset and find that this approach enables solving tasks previously not solvable without explicit chain-of-thought, at a speed comparable to no chain-of-thought."""
#x_test['authors'] = 'Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, Ranjay Krishna'.split(', ')
#x_test['authors'] = 'Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, Stuart Shieber'.split(', ')
#x_test['title'] = 'Stronger Random Baselines for In-Context Learning'
#x_test['abstract'] = 'Evaluating the in-context learning classification performance of language models poses challenges due to small dataset sizes, extensive prompt-selection using the validation set, and intentionally difficult tasks that lead to near-random performance. The standard random baseline -- the expected accuracy of guessing labels uniformly at random -- is stable when the evaluation set is used only once or when the dataset is large. We account for the common practice of validation set reuse and existing small datasets with a stronger random baseline: the expected maximum accuracy across multiple random classifiers. When choosing the best prompt demonstrations across six quantized language models applied to 16 BIG-bench Lite tasks, more than 20\% of the few-shot results that exceed the standard baseline do not exceed this stronger random baseline. When held-out test sets are available, this stronger baseline is also a better predictor of held-out performance than the standard baseline, avoiding unnecessary test set evaluations. This maximum random baseline provides an easily calculated drop-in replacement for the standard baseline. '
#x_test['authors'] = 'Gregory Yauney, David Mimno'.split(', ')
#
#x_test['categories'] = ['cs.CV']
x_test['categories'] = ['cs.CL']
text = convert_to_features(x_test)
print (text)
logits = get_logits(text, tokenizer, model).softmax(dim=-1)
print (logits)
import pdb; pdb.set_trace()
#    title = x['title'].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').strip()
#    authors = x['authors']
#    authors = ', '.join(authors)
#    abstract = x['abstract'].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').strip()
#    published = x['published']
#    day = convert_to_weekday(published)
#    time = published.split('T')[-1]
#    categories = x['categories']

# first, get all logits and labels
all_labels = []
all_logits = []
for x, y in tqdm.tqdm(data['test']):
    text = convert_to_features(x)
    logits = get_logits(text, tokenizer, model).softmax(dim=-1)
    all_logits.append(logits.cpu())
    all_labels.append(1 if y else 0)
all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.LongTensor(all_labels).view(-1)

precisions = []
recalls = []
repeats = 1
ks = list(range(1, 32, 2))          # k values from 1 to 20
ks = np.linspace(0, all_logits.max().item(), num=100)
ks = np.linspace(0, 1.0, num=100)
#import pdb; pdb.set_trace()
for k in ks:
    predictions_all = (all_logits[:, 1] > k).long()
    eval_results = compute_metrics(predictions_all, all_labels)
    precision = eval_results['precision']
    recall = eval_results['recall']
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
#for i, k in enumerate(ks):
#    ax.text(precisions[i]-0.03, recalls[i], f'{k:.2f}', fontsize=5, ha='right', va='center')

# Adding labels and title
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
#ax.set_title('Precision vs Recall for Different k in k-NN')

# Create a ScalarMappable and Colorbar
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])  # You need to set_array([]) to avoid warnings in some Matplotlib versions
cbar = plt.colorbar(sm, ax=ax, aspect=10)  # Link the colorbar to the ax
#cbar.set_label('Value of k')
cbar.set_label('Threshold')

plt.savefig('finetune_deberta_nomultiplier.png')
torch.save({'precisions': precisions, 'recalls': recalls, 'thresholds': ks}, 'finetune_deberta_test.pt')
