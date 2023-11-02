from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
import logging
import argparse
import torch
def get_id2x(input_file):
    with open(input_file) as jf:
        list_base = json.load(jf)
    return list_base

def re_rank(args):
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    model = SentenceTransformer(args.model_path)
    
    list_pid2passage = get_id2x("./data/dataset/corpus_pid2passage.json")
    list_qid2pids = get_id2x(args.id2pids_file)
    list_qid2query = get_id2x(args.id2query_file)    
    all_passage_embedding = None
    
    def similarity_rank(query, passages):
        nonlocal all_passage_embedding
        #Compute embedding for both lists
        query_embedding = model.encode(query, convert_to_tensor=True)
        if args.rerank:
            passages_embeddings = model.encode(passages, show_progress_bar=True, convert_to_tensor=True)
        else:
            if all_passage_embedding is None:
                passages_embeddings = model.encode(passages, show_progress_bar=True, convert_to_tensor=True)
                all_passage_embedding = passages_embeddings
            else:
                passages_embeddings = all_passage_embedding
        
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(query_embedding, passages_embeddings)
        #Output the pairs with their score
        return [score.item() for score in cosine_scores[0]]
    
    def similarity_rank_with_split(query, passages, split_num=2):
        result = []
        #Compute embedding for both lists
        query_embedding = model.encode(query, convert_to_tensor=True)
        for passage in passages:
            passage = [ passage[0:len(passage)//split_num], passage[len(passage)//split_num:-1] ] 
            passage_embeddings = model.encode(passage, convert_to_tensor=True)
            
            cosine_scores = util.cos_sim(query_embedding, passage_embeddings)
            cosine_score = torch.mean(cosine_scores)
            
            result.append(cosine_score.item())
        return result
    
    for qid, query in tqdm(list_qid2query.items(), desc="Ranking"):
        if not args.rerank:
            pids = list(list_pid2passage.keys())
            scores = similarity_rank(query, list(list_pid2passage.values()))
        else:
            pids = list_qid2pids[qid]
            passages = [list_pid2passage[pid] for pid in pids]
            scores = similarity_rank(query, passages)
        # scores = similarity_rank_with_split(query, passages, split_num=2)
        result = [(pid, score) for pid, score in zip(pids, scores)]
        result = sorted(result, key=lambda x: x[1], reverse=True)
        result = result[:100]

        with open(args.output_file, "a", newline='') as f:
            for i, (pid, score) in enumerate(result):
                f.write(f"{qid} Q0 {pid} {i+1} {score:.4f} MPNET\n")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-ranking Script')
    parser.add_argument('--model_path', type=str, default="./model/checkpoint", required=False, help='Path to the pre-trained model')
    parser.add_argument('--id2query_file', type=str, default="./data/dataset/test_qid2query.json",required=False, help='Path to the input file')
    parser.add_argument('--id2pids_file', type=str, default="./data/dataset/test_qid2pids.json", required=False, help='Path to the input file')
    parser.add_argument('--output_file', type=str, default="./result/test_ranking_result_f.txt", required=False, help='Path to the output file')
    parser.add_argument("--log_file", type=str, default="./log/re_ranking.log", help="Path to the log file")
    parser.add_argument("--rerank", type=bool, default=True, help="Whether to re-rank the results"   )
    args = parser.parse_args()
    re_rank(args)