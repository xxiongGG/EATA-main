import random
from os.path import join as opj

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn

from DataLoader.LLM_Dataset import LLMCustomDataset


def get_q_embedding(pre_Q_embed, Q_num, use_pretrain=True, train_embed=False, embed_dim=128):
    if use_pretrain:
        Q_embedding_matrix = nn.Parameter(data=torch.from_numpy(pre_Q_embed).float(), requires_grad=train_embed)
    else:
        Q_embedding_matrix = nn.Parameter(data=torch.randn(Q_num, embed_dim), requires_grad=train_embed)
    return Q_embedding_matrix


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    return pred, truth


def train_test_split(data, split_rate=.9, shuffle=True):
    if shuffle:
        random.shuffle(data)
    x_seqs_size = len(data)
    num_train = int(x_seqs_size * split_rate)
    train_data = data[: num_train]
    test_data = data[num_train:]
    return train_data, test_data


def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.student_id == student_id])
        all_sequences.extend([student_sequence])
    return all_sequences


def parse_student_seq(student):
    seq = student.sort_values('order')
    q = [questions[q] for q in seq.q_id.tolist()]
    a = seq.response
    h = seq.HOT_tag_xx.add(-1)
    t = list(np.add((a * 5).tolist(), h.tolist()))
    return q, t


def sequences2tl(sequences, trgpath):
    with open(trgpath, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')
    f.close()


def get_inputs(sequences, q_matrix, max_step, embed_dim):
    inputs = np.array([])
    targets = np.array([])
    for q_list, a_list in tqdm.tqdm(sequences, 'Convert to input format: '):
        length = len(q_list)
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        student_input = np.zeros(shape=[length + mod, 2 * embed_dim])
        student_target = np.zeros(shape=[length + mod, 1])
        # print(input.shape)
        for i, q_id in enumerate(q_list):
            if a_list[i] - 5 > 0:
                student_input[i][embed_dim:] = q_matrix[q_id]
                student_target[i][0] = a_list[i]
            else:
                student_input[i][:embed_dim] = q_matrix[q_id]
                student_target[i][0] = a_list[i]
            # print(input.shape)
        inputs = np.append(inputs, student_input)
        targets = np.append(targets, student_target)
    inputs = inputs.reshape(-1, max_step, 2 * embed_dim).astype(np.float32)
    targets = targets.reshape(-1, max_step, 1).astype(np.float32)
    return inputs, targets


def get_target(sequences):
    sequences = torch.tensor(sequences)
    print(sequences)
    targets = sequences[:, 1]
    return targets.numpy()


if __name__ == '__main__':
    q_num = 1038
    max_step = 100
    embed_data = np.load(r'../data/q/q_embedding.npy')
    Q_embedding_matrix = get_q_embedding(embed_data, q_num)
    data_path = r'../data/'
    llm_data_path = opj(r'../data/llm', 'seq_{}'.format(max_step))

    data = pd.read_excel(opj(data_path, 'data_input_20231211.xlsx'),
                         usecols=['order', 'student_id', 'q_id', 'response', 'HOT_tag_xx']
                         ).dropna(subset=['HOT_tag_xx'])

    raw_question = data.q_id.unique().tolist()
    raw_question.sort()
    num_question = len(raw_question)
    questions = {p: i for i, p in enumerate(raw_question)}
    print("number of num_questions: %d" % num_question)

    sequences = parse_all_seq(data.student_id.unique())
    print("Student count is:", len(data.student_id.unique()))
    with open(opj(data_path, 'sequences.txt'), 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            # print("seq:", seq)
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')
    train_sequences, test_sequences = train_test_split(sequences)
    print('Train data length is: {}, test data length is: {}.'.format(len(train_sequences), len(test_sequences)))

    sequences2tl(train_sequences, opj(llm_data_path, 'llm_train.txt'))
    sequences2tl(test_sequences, opj(llm_data_path, 'llm_test.txt'))

    llm_train_data, llm_train_target = get_inputs(train_sequences, Q_embedding_matrix, max_step, 64)
    print('LLM train data shape is: {}.'.format(llm_train_data.shape))
    llm_test_data, llm_test_target = get_inputs(test_sequences, Q_embedding_matrix, max_step, 64)
    print('LLM test data shape is: {}.'.format(llm_test_data.shape))

    np.save(opj(llm_data_path, 'llm_train_data.npy'), llm_train_data)
    np.save(opj(llm_data_path, 'llm_test_data.npy'), llm_test_data)

    train_dataset = LLMCustomDataset(llm_train_data, llm_train_target)
    test_dataset = LLMCustomDataset(llm_test_data, llm_test_target)
    torch.save(train_dataset, opj(llm_data_path, 'llm_train_dataset.pth'))
    torch.save(test_dataset, opj(llm_data_path, 'llm_test_dataset.pth'))
    print('Save done!')
