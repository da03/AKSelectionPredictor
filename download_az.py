import feedparser
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils import *
from datetime import datetime, timedelta


def fetch_arxiv_ids(formatted_date):
    url = f"https://huggingface.co/papers?date={formatted_date}"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    soup = BeautifulSoup(response.text, 'html.parser')
    paper_links = soup.find_all('a', href=True)
    paper_ids = set([])
    for link in paper_links:
        href = link['href']
        if '/papers/' in href and len(href.split('/')) == 3:
            paper_id = href.split('/')[-1]
            if '#' in paper_id:
                paper_id = paper_id.split('#')[0].strip()
            paper_ids.add(paper_id)
    
    return sorted(list(paper_ids)), True

## Example usage
#url = "https://huggingface.co/papers?date=2023-05-05"
#arxiv_ids = fetch_arxiv_ids(url)
#print(arxiv_ids)


def lookup_primary_category(arxiv_id):
    query_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    print (query_url)
    response = feedparser.parse(query_url)
    if response.entries:
        primary_category = response.entries[0].get('arxiv_primary_category', {}).get('term')
        return primary_category, True
    else:
        return None, True

database_name = '/root/az/az.db'
table_name = 'az'
create_database(database_name, table_name)

start_date = datetime.strptime('2023-05-05', '%Y-%m-%d')
end_date = datetime.strptime('2024-04-20', '%Y-%m-%d')

current_date = start_date
delta = timedelta(days=1)  # Increment by one day
ls = []
from collections import Counter
counter = Counter()
total = 0

all_ids = []
while current_date <= end_date:
    formatted_date = current_date.strftime('%Y-%m-%d')
    print (formatted_date)

    results, _ = query_f_with_cache(database_name, table_name, fetch_arxiv_ids, formatted_date)
    ls.append(len(results))

    for result in results:
        primary_category, _ = query_f_with_cache(database_name, table_name, lookup_primary_category, result)
        if primary_category is None:
            continue
        counter[primary_category] += 1
        all_ids.append(result)
        total += 1
        print (primary_category)

    print (results)
    current_date += delta
ls = np.array(ls)

for item, c in counter.most_common(100):
    print (item, c/total)
print (sum(ls) / len(ls))
for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    print (f'{p} percentile: {np.percentile(ls, p)}')

print (np.std(ls), len(ls))
print (len(all_ids), len(set(all_ids)))
