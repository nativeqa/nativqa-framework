import time
import optparse
import os
import sys
import json
import csv
import logging
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import hashlib
from urllib.parse import urlparse
from urllib.parse import parse_qs
from serpapi import GoogleSearch
from shutil import copyfile
from . import __version__

from .utils import (read_seed_queries,
                    ensure_directory,
                    read_failed_data,
                    read_summary_data,
                    write_init_summary,
                    write_file,
                    write_csv_file,
                    read_completed_data,
                    read_txt_data,
                    write_txt_file,
                    read_json_data)


urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(filename)s:%(lineno)s - %(message)s',
                    encoding='utf-8', level=logging.DEBUG)


def extract_completed_img_vid_queries(output_dir):
    output = []
    completed_queries = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith("original_response.json"):
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                for result in data:
                    if 'search_parameters' in result:
                        completed_queries.append(result['search_parameters']['q'])

    output_file = os.path.join(output_dir, 'completed_queries.txt')
    logger.info(f'Total completed queries: {len(completed_queries)}')
    logger.info(f'writing completed queries to: {output_file}')
    write_txt_file(output_file, completed_queries)
    return output_file


def extract_completed_queries(output_dir):
    output = []
    completed_queries = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith("all_related_question_answers.tsv"):
                data = read_completed_data(file_path)
                output.extend(data)
    for row in output:
        if row[2].strip() not in completed_queries:
            completed_queries.append(row[2].strip())

    output_file = os.path.join(output_dir, 'completed_queries.txt')
    logger.info(f'Total completed queries: {len(completed_queries)}')
    logger.info(f'writing completed queries to: {output_file}')
    write_txt_file(output_file, completed_queries)
    return output_file

def video_scrape(engine, data, location, gl, summary_writer, failed_writer, mc=None):
    # engine = 'google_videos'
    search_params = {
        "engine": engine,
        "q": "",
        "location": location,
        "gl": gl,
        # "safe": "active",
        # "cr": "countryJO|countryEG|countryPS|countryMA|countryLB|countryAE|countryKW|countrySA",
        "api_key": os.getenv('API_KEY')
    }
    # if mc is not None:
    #     search_params['cr'] = mc
    # print(data)
    for example in tqdm(data, desc="API request processing"):
        category, query = example[0], example[1]
        search_params['q'] = query.strip()
        try:
            search = GoogleSearch(search_params)
            response = search.get_dict()
            # output_response.append(response)
            # print(response)
            response['search_parameters']['category'] = category
            suggested_srch = []
            if 'suggested_searches' in response:
                for entry in response['suggested_searches']:
                    hash_str = entry['link']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    parsed_url = urlparse(entry['link'])
                    query = parse_qs(parsed_url.query)['q'][0]
                    entry['query'] = query
                    entry['category'] = category
                    suggested_srch.append(entry)
                response['suggested_searches'] = suggested_srch
            rel_resp = []
            if 'related_searches' in response:
                for entry in response['related_searches']:
                    hash_str = entry['link']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    entry['category'] = category
                    rel_resp.append(entry)
                response['related_searches'] = rel_resp
            img_resp = []
            if 'video_results' in response:
                for entry in response['video_results']:
                    hash_str = entry['link']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    entry['category'] = category
                    img_resp.append(entry)
                response['video_results'] = img_resp
            summary_writer.write(f"{json.dumps(response, ensure_ascii=False)}\n")
        except Exception as e:
            logger.error(e)
            entry = {"query": query, "category": category, "Error": str(e)}
            failed_writer.write(f"{json.dumps(entry, ensure_ascii=False)}\n")


def image_scrape(engine, data, location, gl, summary_writer, failed_writer, mc=None):
    # engine = 'google_images'
    search_params = {
        "engine": engine,
        "q": "",
        "location": location,
        "gl": gl,
        "safe": "active",
        # "cr": "countryJO|countryEG|countryPS|countryMA|countryLB|countryAE|countryKW|countrySA",
        "api_key": os.getenv('API_KEY')
    }
    if mc is not None:
        search_params['cr'] = mc
    for example in tqdm(data, desc="API request processing"):
        category, query = example[0], example[1]
        search_params['q'] = query.strip()
        try:
            search = GoogleSearch(search_params)
            response = search.get_dict()
            # output_response.append(response)
            response['search_parameters']['category'] = category
            suggested_srch = []
            if 'suggested_searches' in response:
                for entry in response['suggested_searches']:
                    hash_str = entry['link']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    parsed_url = urlparse(entry['link'])
                    query = parse_qs(parsed_url.query)['q'][0]
                    entry['query'] = query
                    entry['category'] = category
                    suggested_srch.append(entry)
                response['suggested_searches'] = suggested_srch
            rel_resp = []
            if 'related_searches' in response:
                for entry in response['related_searches']:
                    hash_str = entry['link']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    entry['category'] = category
                    rel_resp.append(entry)
                response['related_searches'] = rel_resp
            img_resp = []
            if 'images_results' in response:
                for entry in response['images_results']:
                    hash_str = entry['original']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    entry['category'] = category
                    img_resp.append(entry)
                response['images_results'] = img_resp
            summary_writer.write(f"{json.dumps(response, ensure_ascii=False)}\n")
        except Exception as e:
            logger.error(e)
            entry = {"query": query, "category": category, "Error": str(e)}
            failed_writer.write(f"{json.dumps(entry, ensure_ascii=False)}\n")


def scrape(engine, data, location, gl, summary_writer, failed_writer, mc=None):
    search_params = {
        "engine": engine,
        "q": "",
        "location": location,
        "gl": gl,
        "safe": "active",
        # "cr": "countryJO|countryEG|countryPS|countryMA|countryLB|countryAE|countryKW|countrySA",
        "api_key": os.getenv('API_KEY')
    }
    # output_response = []
    if mc is not None:
        search_params['cr'] = mc
    for example in tqdm(data, desc="API request processing"):
        category, query = example[0], example[1]
        search_params['q'] = query.strip()
        try:
            search = GoogleSearch(search_params)
            response = search.get_dict()
            # output_response.append(response)
            response['search_parameters']['category'] = category
            resp = []
            if 'related_questions' in response:
                for entry in response['related_questions']:
                    hash_str = response['search_parameters']['q'] + " " + entry['question']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    resp.append(entry)
                response['related_questions'] = resp
            qa_resp = []
            if 'questions_and_answers' in response:
                for entry in response['questions_and_answers']:
                    hash_str = response['search_parameters']['q'] + " " + entry['question']
                    hash_id = hashlib.md5(hash_str.encode())
                    entry['data_id'] = hash_id.hexdigest()
                    qa_resp.append(entry)
                response['questions_and_answers'] = qa_resp
            summary_writer.write(f"{json.dumps(response, ensure_ascii=False)}\n")
        except Exception as e:
            logger.error(e)
            entry = {"query": query, "category": category, "Error": str(e)}
            failed_writer.write(f"{json.dumps(entry, ensure_ascii=False)}\n")



def gen_img_vid_output_files(working_dir, summary, search_type="image"):
    output_data = read_summary_data(summary)
    out_file = os.path.join(working_dir, 'original_response.json')
    logger.info(f'writing response to: {out_file}...')
    write_file(out_file, output_data)
    rel_search = []
    sug_search = []
    img_result = []
    for results in output_data:
        category = results['search_parameters']['category']
        query = results['search_parameters']['q']
        if 'related_searches' in results:
            for entry in results['related_searches']:
                rel_search.append(entry)
        if 'suggested_searches' in results:
            for entry in results['suggested_searches']:
                sug_search.append(entry)
        if search_type == "image":
            if 'images_results' in results:
                for entry in results['images_results']:
                    entry['category'] = category
                    entry['input_query'] = query
                    img_result.append(entry)
        else:
            if "video_results" in results:
                for entry in results['video_results']:
                    entry['category'] = category
                    entry['input_query'] = query
                    img_result.append(entry)

    out_file = os.path.join(working_dir, 'related_search.json')
    logger.info(f'writing related questions to: {out_file}...')
    write_file(out_file, rel_search)
    out_file = os.path.join(working_dir, 'suggested_search.json')
    logger.info(f'writing questions answers to: {out_file}...')
    write_file(out_file, sug_search)
    out_file = os.path.join(working_dir, f'{search_type}_results.json')
    logger.info(f'writing questions answers to: {out_file}...')
    write_file(out_file, img_result)


def generate_output_files(working_dir, summary):
    output_data = read_summary_data(summary)
    out_file = os.path.join(working_dir, 'original_response.json')
    logger.info(f'writing response to: {out_file}...')
    write_file(out_file, output_data)
    # category = get_category(summary)
    header = ['data_id', 'category', 'input_query', 'question', 'answer', 'question_type', 'answer_URLs']
    rqa_resp = [header]
    rquestion_resp = []
    qa_response = []
    rsearch_resp = [['category', 'seed_query', 'related_query']]

    for results in output_data:
        null_entry = True
        category = results['search_parameters']['category']
        if 'related_searches' in results:
            for relq in results['related_searches']:
                if 'query' in relq:
                    rsearch_resp.append([category, results['search_parameters']["q"], relq['query']])
        if 'related_questions' in results:
            null_entry = False
            for entry in results['related_questions']:
                ans = ''
                if 'snippet' in entry:
                    ans = entry['snippet']
                elif 'list' in entry:
                    ans = "\n".join(entry['list'])
                if "link" in entry:
                    rqa_resp.append([entry['data_id'], category, results['search_parameters']["q"], entry['question'], ans,
                                     'related_questions', entry['link']])
            rq = {'search_parameters': results['search_parameters'], 'related_questions': results['related_questions']}
            rquestion_resp.append(rq)
        if 'questions_and_answers' in results:
            null_entry = False
            for entry in results['related_questions']:
                # print(entry)
                if 'answer' in entry:
                    rqa_resp.append(
                        [entry['data_id'], category, results['search_parameters']["q"], entry['question'],
                         entry['answer'], 'questions_and_answers', entry['link']])
            rq = {'search_parameters': results['search_parameters'],
                  'questions_and_answers': results['questions_and_answers']}
            qa_response.append(rq)
        if null_entry:
            if 'search_parameters' in results:
                rqa_resp.append([None, category, results['search_parameters']["q"], 'Not Available', 'NA', 'NA', 'NA'])

    out_file = os.path.join(working_dir, 'related_questions.json')
    logger.info(f'writing related questions to: {out_file}...')
    write_file(out_file, rquestion_resp)
    out_file = os.path.join(working_dir, 'questions_answers.json')
    logger.info(f'writing questions answers to: {out_file}...')
    write_file(out_file, qa_response)

    out_file = os.path.join(working_dir, 'all_related_question_answers.tsv')
    logger.info(f'writing all related question answers to: {out_file}...')
    write_csv_file(out_file, rqa_resp)
    # unique_qa = [rqa_resp[0]]
    # temp = []
    # for row in rqa_resp[1:]:
    #     if row[3].strip() not in temp and row[3] !='NA':
    #         temp.append(row[3].strip())
    #         unique_qa.append(row)
    # out_file = working_dir + f'/{folder_name}_unique_questions.tsv'
    # print(f'writing response to: {out_file}...')
    # write_csv_file(out_file, unique_qa, delim='\t')

    out_file = os.path.join(working_dir, 'related_search.tsv')
    logger.info(f'writing related search to: {out_file}...')
    write_csv_file(out_file, rsearch_resp)


def run_nativqa(engine, search_type, input_file, gl, location, multiple_country, result_dir, env, n_iter):
    accepted_input_file = ['csv', 'tsv']
    accepted_search_engine = ['google', 'yahoo', 'bing']
    if engine.lower() in accepted_search_engine:
        engine = engine.lower()
    else:
        logger.error('please specify search engine either `google`, `yahoo`, or `bing`')
        sys.exit(1)
    if search_type.lower() not in ['text', 'image', 'video']:
        logger.error('Invalid search type. Supported search types are [image, text, or video]')
        sys.exit(1)
    if search_type.lower() == 'image':
        if engine.lower() == 'google':
            engine = 'google_images'
        elif engine.lower() == 'bing':
            engine = 'bing_images'
        else:
            logger.error('Only Google and Bing Image search is supported for now!')
            sys.exit(1)
    if search_type.lower() == 'video':
        if engine.lower() == 'google':
            engine = 'google_videos'
        else:
            logger.error('Only Google Image search is supported for now!')
            sys.exit(1)

    if input_file is None:
        logger.error('input file is required!')
        sys.exit(1)
    else:
        extension = input_file.rsplit('.')[-1]
        if extension not in accepted_input_file:
            logger.error(f'The input file is not supported by NativQA Framework.')
            sys.exit(1)
    if n_iter < 1:
        logger.error('Number of iteration should be more than 1.')
        logger.info('The number of iteration set to default value 3.')
    if env is None:
        logger.error('API key file is required to use SerpApi!')
        sys.exit(1)
    else:
        load_dotenv(env)
        if os.getenv('API_KEY') is None:
            logger.error('API_KEY not found in the system environment!')
            sys.exit(1)
    folder_name = os.path.splitext(os.path.basename(input_file))[0]
    if result_dir is None:
        result_dir = f'./results/{search_type}/'+ folder_name
    else:
        result_dir = os.path.join(result_dir, search_type, folder_name)
    ensure_directory(result_dir)

    working_dir = os.path.join(result_dir, 'iteration_1')
    ensure_directory(working_dir)
    output_dir = os.path.join(working_dir, 'output')
    ensure_directory(output_dir)

    query_file = os.path.join(working_dir, os.path.basename(input_file))
    if not os.path.exists(query_file):
        logger.info(f'Copying file to {working_dir}')
        copyfile(input_file, query_file)

    for iteration in range(n_iter):
        # working_dir = os.path.join(result_dir, f'iteration_{iteration+1}')
        # ensure_directory(working_dir)
        # output_dir = os.path.join(working_dir, 'output')
        # ensure_directory(output_directory)

        summary = os.path.join(output_dir, 'summary.jsonl')
        failed = os.path.join(output_dir, 'failed.jsonl')

        if not os.path.exists(failed):
            check_cache = False
        else:
            check_cache = True
        failed_data = []
        if check_cache:
            failed_data = read_failed_data(failed)

        # get summary
        summary_data = []
        if os.path.exists(summary):
            summary_data = read_summary_data(summary)
        summary_writer = open(summary, 'w', encoding='utf-8')
        if len(summary_data) > 0:
            logger.info('writing initial summary')
            summary_writer = write_init_summary(summary_writer, summary_data)
        failed_writer = open(failed, 'w', encoding='utf-8')
        # read data
        logger.info(f'reading file: {query_file}...')
        data = read_seed_queries(query_file)
        if len(failed_data) > 0:
            if len(failed_data) + len(summary_data) != len(data):
                # print("scrapping both failed and rest data")
                logger.info(f'Skipping total data: {len(summary_data)}')
                # print("failed, continuing from failed first followed by summary")
                # first try to scrape failed data, then try to scrape rest
                if search_type == 'text':
                    scrape(engine, failed_data, location, gl, summary_writer, failed_writer, multiple_country)
                elif search_type == 'video':
                    video_scrape(engine, failed_data, location, gl, summary_writer, failed_writer, multiple_country)
                else:
                    image_scrape(engine, failed_data, location, gl, summary_writer, failed_writer, multiple_country)
                start_index = len(failed_data) + len(summary_data)
                data = data[start_index:]
                if search_type == 'text':
                    scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
                elif search_type == 'video':
                    video_scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
                else:
                    image_scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
            else:
                # scrape only failed data
                # print("scrapping only failed data")
                if search_type == 'text':
                    scrape(engine, failed_data, location, gl, summary_writer, failed_writer, multiple_country)
                elif search_type == 'video':
                    video_scrape(engine, failed_data, location, gl, summary_writer, failed_writer, multiple_country)
                else:
                    image_scrape(engine, failed_data, location, gl, summary_writer, failed_writer, multiple_country)
        else:
            # no failed data, need to check summary data
            if len(summary_data) == len(data):
                logger.info('All the data is scraped!')
            elif len(summary_data) > 0:
                # print("No failed, only continuing from summary")
                logger.info(f'Skipping total data: {len(summary_data)}')
                data = data[len(summary_data):]
                if search_type == 'text':
                    scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
                elif search_type == 'video':
                    video_scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
                else:
                    image_scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
            else:
                # starting from input file
                # print("starting from input file")
                if search_type == 'text':
                    scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
                elif search_type == 'video':
                    video_scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
                else:
                    image_scrape(engine, data, location, gl, summary_writer, failed_writer, multiple_country)
        summary_writer.close()
        failed_writer.close()
        if search_type == 'text':
            generate_output_files(output_dir, summary)
            completed_query_file = extract_completed_queries(result_dir)
        else:
            gen_img_vid_output_files(output_dir, summary, search_type)
            completed_query_file = extract_completed_img_vid_queries(result_dir)

        if iteration < (n_iter - 1):
            working_dir = os.path.join(result_dir, f'iteration_{iteration + 2}')
            ensure_directory(working_dir)
            query_file = os.path.join(working_dir, os.path.basename(input_file))
            output_data = [['topic', 'query']]
            completed_query = read_txt_data(completed_query_file)
            if search_type == 'text':
                result_file = os.path.join(output_dir, 'all_related_question_answers.tsv')
                search_result = read_completed_data(result_file)
                for row in search_result:
                    if row[3].strip() == 'Not Available':
                        continue
                    if row[3].strip() not in completed_query and row[3].strip() not in output_data:
                        output_data.append([row[1].strip(), row[3].strip()])

                rs_file = os.path.join(output_dir, 'related_search.tsv')
                rsearch_data = read_completed_data(rs_file)
                for row in rsearch_data:
                    if row[2].strip() == 'Not Available':
                        continue
                    if row[2].strip() not in completed_query and row[2].strip() not in output_data:
                        output_data.append([row[0].strip(), row[2].strip()])
            else:
                rs_file = os.path.join(output_dir, 'related_search.json')
                sug_file = os.path.join(output_dir, 'suggested_search.json')
                rs_data = read_json_data(rs_file)
                sug_data = read_json_data(sug_file)
                for obj in rs_data:
                    output_data.append([obj['category'], obj['query']])
                for obj in sug_data:
                    output_data.append([obj['category'], obj['query']])
            write_csv_file(query_file, output_data)
            output_dir = os.path.join(working_dir, 'output')
            ensure_directory(output_dir)

    # merge data and consolidate final QA pair
    if search_type == 'text':
        header = ['data_id', 'category', 'input_query', 'question', 'answer', 'question_type', 'answer_URLs']
        output = []
        filtered_output = [header]

        duplicate = [header]

        for root, dirs, files in os.walk(result_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith("all_related_question_answers.tsv"):
                    data = read_completed_data(file_path)
                    output.extend(data)

        for row in output:
            flag = False
            for data in filtered_output:
                if row[3].strip() == data[3].strip() and row[4].strip() == data[4].strip():
                    flag = True
                if row[3] == 'Not Available' or row[4] == 'NA':
                    flag = True
            if not flag:
                filtered_output.append(row)
            else:
                duplicate.append(row)
    else:
        filtered_output = []
        ids = []
        duplicate = []
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(f"{search_type}_results.json"):
                    data = read_json_data(file_path)
                    for obj in data:
                        if obj['data_id'] not in ids:
                            ids.append(obj['data_id'])
                            filtered_output.append(obj)
                        else:
                            duplicate.append(obj)

    dataset_dir = os.path.join(result_dir, 'dataset')
    ensure_directory(dataset_dir)
    if search_type == 'text':
        dataset_file = os.path.join(dataset_dir, f'{folder_name}.tsv')
        duplicate_file = os.path.join(dataset_dir, f'{folder_name}.duplicate_qa.tsv')
        logger.info(f'Total unique data collected: {len(filtered_output)}')
        logger.info(f'Total duplicate data collected: {len(duplicate)}')
        logger.info(f'writing output to: {dataset_file}')
        write_csv_file(dataset_file, filtered_output)
        logger.info(f'writing duplicate to: {duplicate_file}')
        write_csv_file(duplicate_file, duplicate)
    else:
        dataset_file = os.path.join(dataset_dir, f'{folder_name}.json')
        duplicate_file = os.path.join(dataset_dir, f'{folder_name}.duplicate_qa.json')
        logger.info(f'Total unique data collected: {len(filtered_output)}')
        logger.info(f'Total duplicate data collected: {len(duplicate)}')
        logger.info(f'writing output to: {dataset_file}')
        write_file(dataset_file, filtered_output)
        logger.info(f'writing duplicate to: {duplicate_file}')
        write_file(duplicate_file, duplicate)

def main():
    parser = optparse.OptionParser(version=f'%prog {__version__}')
    parser.add_option('-s', '--engine', action="store", dest="engine", default=None, type="string",
                      help='Search engine (google, yahoo, or bing)')
    parser.add_option('-t', '--search_type', action="store", dest="search_type", default="text", type="string",
                      help="Type of search API (image, text, or video)")
    parser.add_option('-i', '--input_file', action="store", dest="input_file", default=None, type="string",
                      help='input csv/tsv file to scrape')
    parser.add_option('-c', '--country_code', action='store', dest="country_code", default="qa", type="string",
                      help="Country code supported by Google")
    parser.add_option('-l', '--location', action='store', dest="location", default="Doha, Qatar", type="string",
                      help="Location supported by Google")
    parser.add_option('-m', '--multiple_countries', action='store', dest="multiple_countries", default=None,
                      type="string",
                      help="Multiple countries, supported by Google")
    parser.add_option('-o', '--output_dir', action="store", dest="output_dir", default=None, type='string',
                      help="output directory location")
    parser.add_option('-e', '--env', action='store', dest='env', default=None, type="string",
                      help='API key file')
    parser.add_option('-n', '--n_iter', action='store', dest='n_iter', default=3, type="int",
                      help='Number of iteration for data scrape')

    options, args = parser.parse_args()
    engine = options.engine
    search_type = options.search_type
    input_file = options.input_file
    gl = options.country_code
    location = options.location
    env = options.env
    result_dir = options.output_dir
    n_iter = options.n_iter
    multiple_country = options.multiple_countries
    run_nativqa(engine, search_type, input_file, gl, location, multiple_country, result_dir, env, n_iter)

if __name__=="__main__":
    main()
