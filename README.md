# IR HOMEWORK
## trec-dl-2022-passage-rerank

| model                        | ndcg@5     | ndcg@10    | ndcg@20    | ndcg@30    | ndcg@100   |
| --------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| all-mpnet-base-v2           | 0.5131     | 0.4581     | 0.3988     | 0.3653     | 0.2629     |
| all-mpnet-base-v2-finetuned | 0.5043     | 0.4594     | 0.3991     | 0.3657     | 0.2632     |
| bge-large-en-v1.5           | 0.5429      | 0.4881    | 0.4208    | 0.3855        | 0.2726 |
| bge-large-en-v1.5-finetuned | 0.4998     | 0.4453     | 0.3926     | 0.3660     | 0.2640     |
| bge-reranker-large          | **0.5798** | **0.5130** | **0.4452**  | **0.4054**  | **0.2790** |

### QuickStart
1. Download the `all-mpnet-base-v2` / `bge-large-en-v1.5` or `bge-reranker-large` model from huggingface
2. Config the `run.sh` script
3. `cd TREC2022-passage_rerank`  and `./run.sh`
