# MohammedDhiyaEddine/emploitic-sentence-transformer-tsdae-camembert-base
# Evaluating the embeddings calculated by the emploitic-sentence-transformer-tsdae-camembert-base

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# model_folder = './cache/emploitic-tsdae'
# MohammedDhiyaEddine/tsdae-distiluse-roberta-alldata

tokenizer = AutoTokenizer.from_pretrained("MohammedDhiyaEddine/tsdae-distiluse-roberta-alldata")
model = SentenceTransformer("MohammedDhiyaEddine/tsdae-distiluse-roberta-alldata")

TOP_K = 10

EVAL_DATA_FILE = '../data/eval_data_emploitic.tsv'
ENTITIES_FILE = '../data/entities_emploitic.txt'
LOGS_FILE = '../output/tsdae-emploitic/distiluse-roberta/all/logs_k'+ str(TOP_K) +'.txt'
RANKS_FILE = '../output/tsdae-emploitic/distiluse-roberta/all/ranks_k'+ str(TOP_K) +'.txt'

import time
start = time.time()
evaluation_score = 0

# get all eval pairs
df = pd.read_csv(EVAL_DATA_FILE, delimiter='\t', names=['head', 'tail'])
number_of_eval_pairs = len(df.index)
# get all entities from entities.txt
df_entities = pd.read_csv(ENTITIES_FILE, delimiter='\t', names=['entity'])

# group lines by head
df_grouped = df.groupby('head')
# open LOGS_FILE and RANKS_FILE for writing
with open(LOGS_FILE, 'a') as f, open(RANKS_FILE, 'a') as g:
    f.write("Evaluating the embeddings calculated by the emploitic-sentence-transformer-tsdae-camembert-base\r ")
    f.write("cosine similarity\r ")
    f.write("TOP_K = {}\r ".format(TOP_K))
    f.write("Date and time: {}\r ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    entity_embeddings = {}
    for entity in df_entities['entity'].tolist() :
        # get embedding for entity
        emb2 = model.encode(entity)
        entity_embeddings[entity] = emb2
        if(len(entity_embeddings) % 10 == 0):
            print("Embedding entities: {} / {}".format(len(entity_embeddings), len(df_entities['entity'].tolist())))

    heads_done = 0
    pairs_done = 0
    # iterate over groups
    for head, group in df_grouped:
        scores = []
        emb1 = model.encode(head)
        # gather scores for head with each entity
        for entity in df_entities['entity'].tolist() :
            # print(head, entity)
            # get embedding for entity
            emb2 = entity_embeddings[entity]
            sim = cosine_similarity([emb1], [emb2])
            score = sim.item()
            # save head, entity, score to file no
            f.write("{}\t{}\t{:.2f}\r ".format(head, entity, score))
            # print(score)
            scores.append(score)
            
        scores.sort(reverse=True)

        # get the list of eval tails
        tails = group['tail'].tolist()
        for tail in tails : 
            print(head, tail)
            g.write("{}\t{}\r ".format(head, tail))
            # predict the score
            emb2 = model.encode(tail)
            sim = cosine_similarity([emb1], [emb2])
            score = sim.item()
            # ensure that the score is in the list
            scores.append(score)
            scores.sort(reverse=True)
            # get rank of the score
            rank = scores.index(score)
            print(score, rank)
            g.write("{:.2f}\t{}\r ".format(score, rank))
            f.write("{}\t{}\t{:.2f}\t{}\r ".format(head, tail, score, rank))
            # add 1 to the evaluation score if the rank is less than TOP_K
            if rank < TOP_K :
                evaluation_score += 1
            pairs_done += 1

        heads_done += 1
        progress = heads_done * 100 / len(df_grouped.groups.keys())
        print("Progress: {:.2f}%".format(progress))
        current_evaluation_score = evaluation_score * 100 / (number_of_eval_pairs)
        print("Current evaluation score: {:.2f}%".format(current_evaluation_score))
            

    elapsed = time.time() - start
    print("Elapsed time:", elapsed) 
    print("Pairs done:", pairs_done)
    f.write("Elapsed time: {}\r ".format(elapsed))
    g.write("Elapsed time: {}\r ".format(elapsed))
    print("Evaluation score:", evaluation_score)
    f.write("Evaluation score: {}\r ".format(evaluation_score))
    g.write("Evaluation score: {}\r ".format(evaluation_score))
    print("Evaluation score normalized:", evaluation_score * 100 / (number_of_eval_pairs))
    f.write("Evaluation score normalized: {}\r ".format(evaluation_score * 100 / (number_of_eval_pairs)))
    g.write("Evaluation score normalized: {}\r ".format(evaluation_score * 100 / (number_of_eval_pairs)))



