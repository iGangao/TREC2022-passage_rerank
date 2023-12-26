# -*- encoding:utf-8 -*-
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from datetime import datetime
from utils import InformationRetrievalEvaluator
import sys
import os
import random
from loguru import logger
import json
from tqdm import tqdm
import argparse
class Train(object):
    def __init__(self,args):
        self.prefix_path = args.prefix_path
        self.model_name = args.model_name
        self.pretrained_model_path = args.pretrained_model_path
        self.train_batch_size = args.train_batch_size
        self.max_seq_length = args.max_seq_length
        self.num_epochs = args.num_epochs
        self.lr=args.learning_rate
        self.weight_decay=args.weight_decay
        self.dev_data_path = args.dev_data_path
        self.model_save_path = args.model_save_path

    def __load_map(self, ):
        
        train_qid2pid, train_id2query, corpus_id2passage, val_qid2query, val_qid2pids = [
            json.load(open(os.path.join(self.prefix_path, file_name),'r')) for file_name in \
                ['train_qid2pid_rate.json', 'train_id2query.json', 
                'corpus_pid2passage.json', 'val_id2query.json', 
                'val_qid2pids.json']]
        
        return train_qid2pid, train_id2query, corpus_id2passage, val_qid2query, val_qid2pids
    
    def __build_model(self, ): 
        word_embedding_model = models.Transformer(self.pretrained_model_path, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda")
        # model = torch.nn.DataParallel(model).to("cuda")
        return model

    def __build_train_dataloader(self, train_id2query, corpus_id2passage, train_qid2pid):
        train_samples = []
        for key in tqdm(train_qid2pid.keys()):
            query = train_id2query[key]
            pos_passage = corpus_id2passage[train_qid2pid[key]]
            neg_passage = corpus_id2passage[random.choice([pid for pid in corpus_id2passage.keys() if pid!=key])]
            train_samples.append(InputExample(texts=[query, pos_passage, neg_passage]))
        
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=self.train_batch_size)
        return train_dataloader

    def __build_dev_evaluator(self, val_qid2query, val_qid2pids, corpus_id2passage):
        dev_data = open(self.dev_data_path, 'r')
        qid2query, pid2passage, relevant_docs = val_qid2query, corpus_id2passage, dict()
        for line in tqdm(dev_data.readlines()):
            try:
                qid, split_signal, pid, score = line.strip().split()
                if qid not in relevant_docs.keys():
                    relevant_docs[qid] = dict()
                relevant_docs[qid][pid] = float(score)
            except KeyError:
                continue
        dev_evaluator = InformationRetrievalEvaluator(queries=qid2query, corpus=pid2passage, 
                                                      init_top100=None, relevant_docs=relevant_docs, 
                                                      ndcg_at_k=[10], name='dev', show_progress_bar=True, )
        return dev_evaluator
    
    def __call__(self, ):
        logger.info('loading map ...')
        train_qid2pid, train_id2query, corpus_id2passage, val_qid2query, val_qid2pids = self.__load_map()
        
        logger.info('building model ...')
        model = self.__build_model()
        
        logger.info('building train dataloader ...')
        train_dataloader = self.__build_train_dataloader(train_id2query, corpus_id2passage, train_qid2pid)
        
        logger.info('building dev dataloader ...')
        dev_evaluator = self.__build_dev_evaluator(val_qid2query, val_qid2pids, corpus_id2passage)
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)
        evaluation_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)

        logger.info("Warmup-steps: {}".format(warmup_steps))
        
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=dev_evaluator,
                epochs=self.num_epochs,
                steps_per_epoch=int(len(train_dataloader))//self.train_batch_size,
                evaluation_steps=evaluation_steps,
                optimizer_params = {'lr': self.lr},
                weight_decay=self.weight_decay,
                warmup_steps=warmup_steps,
                output_path=self.model_save_path,
                )
        logger.info("finish training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', default="../data/dataset", type=str)
    parser.add_argument('--model_name', default="all-mpnet-base-v2", type=str)
    parser.add_argument('--pretrained_model_path', default="../model/all-mpnet-base-v2", type=str)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--max_seq_length', default=384, type=int)
    parser.add_argument('--num_epochs', default=15, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--dev_data_path', default="../data/row-dataset/val_2021.qrels.pass.final.txt" , type=str)
    parser.add_argument('--model_save_path', default="../model/all-mpnet-base-v2-finetuned", type=str)
    args = parser.parse_args()
    t = Train(args)
    sys.exit(t())