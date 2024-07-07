from utils import *
from datetime import datetime, timedelta

az_database_name = '/root/az/az.db'
az_table_name = 'az'

start_date = datetime.strptime('2023-05-05', '%Y-%m-%d')
end_date = datetime.strptime('2024-04-20', '%Y-%m-%d')

current_date = start_date
delta = timedelta(days=1)  # Increment by one day

az_arxiv_ids = set([])
while current_date <= end_date:
    formatted_date = current_date.strftime('%Y-%m-%d')
    results, _ = query_f_with_cache(az_database_name, az_table_name, None, formatted_date)
    for result in results:
        az_arxiv_ids.add(result)
    current_date += delta

arxiv_database_name = '/root/az/arxiv.db'
arxiv_table_name = 'arxiv'


start_date = datetime.strptime('2023-05-05', '%Y-%m-%d')
train_cutoff_date = datetime.strptime('2024-02-20', '%Y-%m-%d')
valid_cutoff_date = datetime.strptime('2024-03-20', '%Y-%m-%d')
end_date = datetime.strptime('2024-04-20', '%Y-%m-%d')


current_date = start_date
delta = timedelta(days=1)  # Increment by one day

results_train = []
results_valid = []
results_test = []


def process_result(result):
    arxiv_id = result['id'].split('/')[-1]
    published = result['published']
    title = result['title']
    if 'v1' in arxiv_id:
        arxiv_id = arxiv_id.replace('v1', '')
    authors = [author['name'] for author in result['authors']]
    abstract = result['summary']

    categories = []
    #import pdb; pdb.set_trace()
    primary_category = result['arxiv_primary_category']['term']
    categories.append(primary_category)
        
    # Get all categories listed for the paper, including cross-listed ones
    all_categories = [tag['term'] for tag in result['tags'] if 'term' in tag]
        
    # Add cross-listed categories, ensuring the primary is not duplicated
    for category in all_categories:
        if category not in categories:
            categories.append(category)
    return {'arxiv_id': arxiv_id,
            'published': published,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'categories': categories}

arxiv_ids = set([])
while current_date <= end_date:
    if current_date <= train_cutoff_date:
        results_all = results_train
    elif current_date <= valid_cutoff_date:
        results_all = results_valid
    else:
        results_all = results_test


    #import pdb; pdb.set_trace()
    formatted_date = current_date.strftime('%Y%m%d')
    for category in ['cs.CV', 'cs.CL', 'cs.LG']:
        print (category, formatted_date)
        results, _ = query_f_with_cache(arxiv_database_name, arxiv_table_name, None, category + ':' + formatted_date)

        for result in results:
            if result['published'] != result['updated']:
                continue
            entry = process_result(result)
            if entry['arxiv_id'] in arxiv_ids:
                continue
            arxiv_ids.add(entry['arxiv_id'])
            results_all.append((entry, entry['arxiv_id'] in az_arxiv_ids))
    #import pdb; pdb.set_trace()
    current_date += delta


import random
random.seed(1234)

for results_all in [results_train, results_valid, results_test]:
    total = 0
    total_selected = 0
    for _, selected in results_all:
        total += 1
        if selected:
            total_selected += 1
    print (total_selected/total)

import torch
torch.save({'train': results_train, 'valid': results_valid, 'test': results_test}, 'dataset.pt')
