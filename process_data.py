from collections import defaultdict
import json
import csv
import os

def convert_tsv_to_json(input_file, output_file):
    csv.field_size_limit(100*1024*1024)
    
    with open(input_file, "r") as f:
        data = csv.reader(f, delimiter="\t")
        dic = {row[0]: row[1] for row in data}

    with open(output_file, "w") as jf:
        json.dump(dic, jf, indent=4)

def convert_txt_to_json(input_file, output_file):    
    with open(input_file, 'r') as txt_file:
        data = [line.strip().split(' ') for line in txt_file]
        dic = defaultdict(list)
        for row in data:
            dic[row[0]].append(row[2])
    with open(output_file, "w") as jf:
        json.dump(dic, jf, indent=4)    

def convert_x_to_json(input_file, output_file, index=1):
    with open(input_file,'r') as f:
        if input_file.endswith('.tsv'):
            data = csv.reader(f, delimiter="\t")
            dic = {row[0]: row[index] for row in data}
        elif input_file.endswith('.txt'):
            data = [line.strip().split(' ') for line in f]
            dic = defaultdict(list)
            for row in data:
                dic[row[0]].append(row[index])
    with open(output_file, "w") as jf:
        json.dump(dic, jf, indent=4)
        

if __name__ == "__main__":
    conversion_tasks = [
        ("./data/row-dataset/collection.sampled.tsv", "./data/dataset/corpus_pid2passage.json", 1),
        ("./data/row-dataset/train_sample_queries.tsv", "./data/dataset/train_id2query.json", 1),
        ("./data/row-dataset/test_2022_76_queries.tsv", "./data/dataset/test_qid2query.json", 1),
        ("./data/row-dataset/val_2021_53_queries.tsv", "./data/dataset/val_id2query.json", 1),
        ("./data/row-dataset/train_sample_passv2_qrels.tsv", "./data/dataset/train_qid2pid_rate.json", 2),
        ('./data/row-dataset/test_2022_passage_top100.txt', "./data/dataset/test_qid2pids.json", 2),
        ('./data/row-dataset/val_2021_passage_top100.txt', "./data/dataset/val_qid2pids.json", 2)
    ]

    for input_path, output_path, index in conversion_tasks:
        if not os.path.exists(output_path):
            convert_x_to_json(input_path, output_path, index)
