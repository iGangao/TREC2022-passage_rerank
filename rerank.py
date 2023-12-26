from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
import logging
import argparse
import torch
from FlagEmbedding import FlagReranker

class Reranker(object):
    def __init__(self, args) -> None:
        self.model_path = args.model_path
        self.model = self.build_model(args.model_path)
        self.list_pid2passage = self.get_id2x(args.corpus_file)  # "./data/dataset/corpus_pid2passage.json"
        self.list_qid2pids = self.get_id2x(args.id2pids_file)
        self.list_qid2query = self.get_id2x(args.id2query_file)
        self.output_file = args.output_file
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    def get_id2x(self, input_file):
        with open(input_file) as jf:
            list_base = json.load(jf)
        return list_base
    
    def compute_scores(self, query, passages):
        if "reranker" in self.model_path:
            texts = []
            for passage in passages:
                texts.append([query, passage])
            scores = self.model.compute_score(texts)
            return scores
        else:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            passages_embeddings = self.model.encode(passages, show_progress_bar=True, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, passages_embeddings)
            return [score.item() for score in cosine_scores[0]]

    def build_model(self, model_path='./model/bge-reranker-large'):
        if "reranker" in model_path:
            return FlagReranker(model_path, use_fp16=True)
        else:
            return SentenceTransformer(model_path)

    def rerank(self):
        for qid, query in tqdm(self.list_qid2query.items(), desc="Ranking"):
            pids = self.list_qid2pids[qid]
            passages = [self.list_pid2passage[pid] for pid in pids]

            scores = self.compute_scores(query, passages)
            
            result = [(pid, score) for pid, score in zip(pids, scores)]
            result = sorted(result, key=lambda x: x[1], reverse=True)
            result = result[:100]

            with open(self.output_file, "a", newline='') as f:
                for i, (pid, score) in enumerate(result):
                    f.write(f"{qid} Q0 {pid} {i+1} {score:.4f} MPNET\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-ranking Script')
    parser.add_argument('--model_path', type=str, default="./model/checkpoint", required=False, help='Path to the pre-trained model')
    parser.add_argument('--corpus_file', type=str, default="./data/dataset/corpus_pid2passage.json",required=False, help='Path to the input file')
    parser.add_argument('--id2query_file', type=str, default="./data/dataset/test_qid2query.json",required=False, help='Path to the input file')
    parser.add_argument('--id2pids_file', type=str, default="./data/dataset/test_qid2pids.json", required=False, help='Path to the input file')
    parser.add_argument('--output_file', type=str, default="./result/test_ranking_result_re.txt", required=False, help='Path to the output file')
    parser.add_argument("--log_file", type=str, default="./log/re_ranking.log", help="Path to the log file")
    args = parser.parse_args()
    reranker = Reranker(args)
    reranker.rerank()