import os
import csv
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(filename)s:%(lineno)s - %(message)s',
                    encoding='utf-8', level=logging.DEBUG)

def read_seed_queries(fpath):
    delim = ',' if fpath.endswith('.csv') else '\t'
    data = []
    unique = []
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delim)
        next(reader)
        for row in reader:
            # topic, query = row[0].strip(), row[1].strip()
            # if topic not in data:
            #     data[topic] = []
            # data[topic].append(query)
            if row[1].strip() not in unique:
                unique.append(row[1].strip())
                data.append([row[0].strip(), row[1].strip()])
    logger.info(f"Number of queries to be searched: {len(data)}")
    return data

def ensure_directory(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory '{path}' was created.")
        except OSError as error:
            logger.error(f"Creation of the directory '{path}' failed due to: {error}")
    else:
        logger.warning(f"Directory '{path}' already exists.")

# def ensure_directory(dir_path):
#     if os.path.exists(dir_path):
#         logging.warning(f"{dir_path} directory exists!")
#     if not os.path.isdir(dir_path):
#         os.makedirs(dir_path, exist_ok=True)
#         logging.info(f'Directry created at {dir_path}')

def read_failed_data(filepath):
    q_texts = []
    with open(filepath, encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    if len(lines) == 1 and lines[0] == '':
        return q_texts
    else:
        for line in lines:
            dict_obj = json.loads(line)
            q_texts.append([dict_obj['category'], dict_obj['query']])
        return q_texts

def read_summary_data(filepath):
    data = []
    with open(filepath) as f:
        lines = f.read().strip().split("\n")
    # print(lines)
    if len(lines) == 1 and lines[0] == '':
        return data
    else:
        for line in lines:
            data.append(json.loads(line))
        return data

def read_completed_data(filepath):
    delim = ',' if filepath.endswith('.csv') else '\t'
    data = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter=delim)
        next(reader)
        for row in reader:
            data.append(row)
    return data

def read_txt_data(filepath):
    queries = open(filepath, 'r', encoding='utf-8').read().strip().split("\n")
    return queries

def read_json_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data

def write_init_summary(filewriter, data):
    for line in data:
        filewriter.write(f"{json.dumps(line, ensure_ascii=False)}\n")
    return filewriter

def write_file(filepath, out_data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False)

def write_csv_file(out_file, rqa_data):
    delim = ',' if out_file.endswith('.csv') else '\t'
    with open(out_file,'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delim)
        for row in rqa_data:
            writer.writerow(row)

def write_txt_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        for query in data:
            f.write(query + "\n")
