import json
import sqlite3
from tenacity import (
retry,
wait_exponential,
wait_fixed,
stop_after_attempt,
)
import hashlib

def compute_metrics(pred_labels, labels):
    # True positives, false positives, true negatives, false negatives
    tp = ((pred_labels == 1) * (labels == 1)).sum().item()
    fp = ((pred_labels == 1) * (labels == 0)).sum().item()
    tn = ((pred_labels == 0) * (labels == 0)).sum().item()
    fn = ((pred_labels == 0) * (labels == 1)).sum().item()
    # Precision, Recall, F1, and Accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

def create_database(database_name, table_name):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    # Create table
    c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}
                 (key TEXT PRIMARY KEY, prompt TEXT, completion TEXT)''')
    conn.commit()
    conn.close()

@retry(wait=wait_fixed(1), stop=stop_after_attempt(7))
def insert_or_update(database_name, table_name, key, prompt, completion):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute(f'''INSERT OR REPLACE INTO {table_name}
                 (key, prompt, completion) VALUES (?, ?, ?)''', 
                 (key, prompt, completion))
    conn.commit()
    conn.close()

def retrieve(database_name, table_name, key):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()

    c.execute(f"SELECT prompt, completion FROM {table_name} WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return (True, result)
    else:
        return (False, None)


def query_f_with_cache(database_name, table_name, f, content):
    key = hashlib.sha256(content.encode('utf-8')).hexdigest()
    hit, result = retrieve(database_name, table_name, key)
    #if hit:
    #    prompt, output = result
    #    output = json.loads(output)
    #    if '_splitted' in output:
    #        hit = False
    #        print ('splitted')
    if not hit:
        output, finished = f(content)
        print ('success')
        if finished:
            insert_or_update(database_name, table_name, key, content, json.dumps(output))
    else:
        #print ('hit')
        #print (content)
        prompt, output = result
        output = json.loads(output)
    return output, hit

#@retry(wait=wait_fixed(10), stop=stop_after_attempt(7))
def query_moderation(content, max_num_retries=2, wait_time=20):
    print ('ca', content[:10], len(content))
    client = OpenAI()
    num_retries = 0
    finished = False
    while (not finished) and num_retries <= max_num_retries:
        if num_retries > 0:
            print (f'retrying {num_retries} times')
        try:
            response = client.moderations.create(input=content)
            finished = True
        except Exception as e:
            #raise e
            err_msg = f'{e}'
            print (err_msg)
            m = re.search(r"Please try again in (\d+\.?\d*)s", err_msg)
            num_retries += 1
            if m:
                sleep_time = min(float(m.group(1)) * 1.2, wait_time)
                sleep_time = wait_time
                print (f'sleeping: {sleep_time}')
                time.sleep(sleep_time)
            else:
                time.sleep(wait_time)
    if not finished:
        #import pdb; pdb.set_trace()
        content_length = len(content)
        half_length = int(round(content_length / 2))
        content_firsthalf = content[:half_length]
        content_secondhalf = content[half_length:]
        print (f'splitting, old length: {content_length}, new length: {half_length}')
        #output_firsthalf = query_moderation(content_firsthalf, max_num_retries, wait_time)
        output_firsthalf, _ = query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content_firsthalf)
        #output_secondhalf = query_moderation(content_secondhalf, max_num_retries, wait_time)
        output_secondhalf, _ = query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content_secondhalf)
        output = {'flagged': output_firsthalf['flagged'] or output_secondhalf['flagged']}
        output['categories'] = {}
        for k in output_firsthalf['categories']:
            output['categories'][k] = output_firsthalf['categories'][k] or output_secondhalf['categories'][k]
        output['category_scores'] = {}
        for k in output_firsthalf['category_scores']:
            output['category_scores'][k] = max(output_firsthalf['category_scores'][k], output_secondhalf['category_scores'][k])
        output['_splitted'] = True
        #import pdb; pdb.set_trace()
    else:
        output = response.results[0].model_dump()
    #return output, finished
    return output, True 

def query_moderation_with_cache_worker(contents):
    i = 0
    contents = list(contents)
    for content in tqdm.tqdm(contents):
        if i % 1000 == 0:
            print (i)
        try:
            query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content)
        except Exception as e:
            print (f'Skipping Error: {e}')
            time.sleep(1)
        i += 1
