import feedparser
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils import *
from datetime import datetime, timedelta
import time


def fetch_arxiv_papers(args, max_results=100):
    category, date = args.split(':')
    base_url = "http://export.arxiv.org/api/query?"
    papers = []
    start = 0  # Start at the beginning of the result set
    while True:
        query = f"cat:{category}+AND+submittedDate:[{date}0000+TO+{date}2359]"
        url = f"{base_url}search_query={query}&start={start}&max_results={max_results}"
        print (url)
        #import pdb; pdb.set_trace()
        response = feedparser.parse(url)
        
        if not response.entries:  # Break the loop if no more entries are returned
            break
        
        papers.extend(response.entries)
        start += len(response.entries)
        
        if len(response.entries) < max_results:
            break  # If less than max_results papers were returned, it's the last page

        time.sleep(3)  # Respect arXiv's rate limit

    return papers, True

database_name = '/root/az/arxiv.db'
table_name = 'arxiv'
create_database(database_name, table_name)

start_date = datetime.strptime('2023-05-05', '%Y-%m-%d')
end_date = datetime.strptime('2024-04-20', '%Y-%m-%d')

for category in ['cs.CV', 'cs.CL', 'cs.LG']:
    current_date = start_date
    delta = timedelta(days=1)  # Increment by one day
    while current_date <= end_date:
        formatted_date = current_date.strftime('%Y%m%d')
        print (category, formatted_date)
        #import pdb; pdb.set_trace()
    
        results, _ = query_f_with_cache(database_name, table_name, fetch_arxiv_papers, category + ':' + formatted_date)
        #print (results)
        current_date += delta
