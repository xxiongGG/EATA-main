import random
from os.path import join as opj

import numpy as np
import pandas as pd
import torch
from text2vec import SentenceModel
from torch import nn
from tqdm import tqdm

from DataLoader.Q_Dataset import QCustomDataset


# Get question difficulty by students' response.
def get_Q_Diff(data, questions, d_level=100):
    q_difficulty = {}
    for q in tqdm(questions):
        q_idxes = data[(data.q_id == q)].index.tolist()
        q_data = data.iloc[q_idxes]
        response = np.array(q_data.response)
        if len(response) == 0 or len(q_idxes) < 10:
            q_difficulty[q] = 0
            continue
        d = int(np.mean(response) * d_level) + 1
        q_difficulty[q] = d
    q_difficulty = list(q_difficulty.values())
    embedding_layer = nn.Embedding(num_embeddings=d_level + 2, embedding_dim=64)
    q_difficulty = embedding_layer(torch.tensor(q_difficulty)).detach()
    print('Q difficulty embedding shape is: {}.'.format(q_difficulty.shape))
    return q_difficulty


# Get question text embedding.
def get_Q_Text(data):
    q_texts = {}
    q_data = data[['q_id', 'q_content']].drop_duplicates(subset='q_id', keep='first').sort_values(by='q_id')
    for q_index, q in q_data.iterrows():
        q = q.tolist()
        if q[0] not in q_texts:
            q_texts[q[0]] = q[1]
    return q_texts


# Get question embedding by Text2Vec and reshape by PCA
# def get_text_embed(sentence_list):
#     model = SentenceModel(r'C:\Users\94207\OneDrive\Files\4-code\1-PretrainedModels\text2vec-base-multilingual')
#     embeddings = model.encode(sentence_list, max_seq_length=256)
#     embeddings = torch.tensor(embeddings).T
#     pca = PCA(n_components=64)
#     pca.fit(embeddings)
#     pca_comp = torch.tensor(np.asarray(pca.components_), dtype=torch.float32).T
#     linear = nn.Linear(pca_comp.shape[1], 64, False)
#     activate_fun = nn.Tanh()
#     pca_embeddings = activate_fun(linear(pca_comp)).detach()
#     print('Q text embedding shape is: {}.'.format(pca_embeddings.shape))
#     return pca_embeddings

def get_text_embed(sentence_list):
    model = SentenceModel(r'C:\Users\94207\OneDrive\Files\4-code\1-PretrainedModels\text2vec-base-multilingual')
    embeddings = model.encode(sentence_list, max_seq_length=256)
    embeddings = torch.tensor(embeddings)
    print('Q text embedding shape is: {}.'.format(embeddings.shape))
    return embeddings


def get_Q_HOTS(data):
    q_HOTS = {}
    q_data = data[['q_id', 'HOT_tag_xx']].drop_duplicates(subset='q_id', keep='first').sort_values(by='q_id')
    for q_index, q in q_data.iterrows():
        q = q.tolist()
        if q[0] not in q_HOTS:
            q_HOTS[q[0]] = q[1] - 1
    return q_HOTS


def get_data_distribution(
        path=r'C:\Users\94207\OneDrive\Files\4-code\2-MyCodes\HOT_Tracing_master_240823\data\data_input_20231211.xlsx'):
    data = pd.read_excel(path, usecols=['HOT_tag_xx'])
    value_counts = pd.DataFrame(data.value_counts()).sort_values(by='HOT_tag_xx')
    value_counts['distribution'] = value_counts['count'] / value_counts['count'].sum()
    distribution = [round(num, 3) for num in list(value_counts['distribution'])]
    gamma = [round(1 - x, 2) for x in distribution]
    print(distribution)
    print(gamma)


if __name__ == '__main__':
    data_path = '../data/q/'
    train_test_rate = 0.9

    data = pd.read_excel(opj(data_path, 'data_input_20231211.xlsx')).dropna(subset=['HOT_tag_xx'])
    questions = set(np.array(data['q_id']))
    q_difficulty = get_Q_Diff(data, questions)
    q_texts = get_Q_Text(data)
    q_embed = get_text_embed(list(q_texts.values()))
    q_HOTS = torch.tensor(list(get_Q_HOTS(data).values()), dtype=torch.float32).unsqueeze(-1)
    # q_features = torch.cat([q_embed, q_difficulty], dim=1)
    # np.save(opj(data_path, 'q_features'), q_features)
    # print('Q features embedding shape is: {}.'.format(q_features.shape))

    q_features = q_embed
    train_rate = int(train_test_rate * q_features.shape[0])
    index_list = list(range(q_features.shape[0]))
    train_index = random.sample(index_list, train_rate)
    test_index = list(set(index_list) - set(train_index))
    train_q_features, train_q_HOTS = q_features[train_index], q_HOTS[train_index]
    test_q_features, test_q_HOTS = q_features[test_index], q_HOTS[test_index]

    train_dataset = QCustomDataset(train_q_features, train_q_HOTS)
    test_dataset = QCustomDataset(test_q_features, test_q_HOTS)
    all_dataset = QCustomDataset(q_features, q_HOTS)

    torch.save(train_dataset, opj(data_path, 'train_q_dataset.pth'))
    torch.save(test_dataset, opj(data_path, 'test_q_dataset.pth'))
    torch.save(all_dataset, opj(data_path, 'all_q_dataset.pth'))

    print('Dataset saved.')
