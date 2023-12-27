import argparse
import json
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from trainer import Trainer, SynonymTrainer
from utils import load_tokenizer

def predict_syn(data_file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model_name_or_path = 'bert-base-chinese'
    args.model_type = 'synonym'
    args.eval_batch_size = 8
    args.max_seq_len = 50
    args.ignore_index = 0
    args.task = 'synonym'
    args.no_cuda = True
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    data = []
    with open(data_file) as f:
        for line in f.readlines():
            data.append(json.loads(line))
        
    all_input_ids, attention_masks, spans, type_tokens = [], [], [], []
    for example in data:
        max_length = args.max_seq_len // 2 - 1
        text1_tokens = tokenizer.tokenize(example['text1'])
        text1_tokens = text1_tokens[:max_length]
        span = [example['span1'][0] + 1, example['span1'][1] + 1]
        text2_tokens = tokenizer.tokenize(example['text2'])
        spans.append(span + [example['span2'][0] + len(text1_tokens) + 2, example['span2'][1] + len(text1_tokens) + 2])
        text2_tokens = text2_tokens[:max_length]
        tokens = [tokenizer.cls_token] + text1_tokens + [tokenizer.sep_token] + text2_tokens
        token_type_ids = [1] * (len(text1_tokens) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + [0] * (args.max_seq_len - len(token_type_ids))
        assert len(input_ids) == args.max_seq_len, "Error with input length {} vs {}".format(len(input_ids), args.max_seq_len)
        assert len(attention_mask) == args.max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), args.max_seq_len)
        assert len(token_type_ids) == args.max_seq_len, "Error with token_type_ids length {} vs {}".format(len(token_type_ids), args.max_seq_len)

        attention_masks.append(attention_mask)
        all_input_ids.append(input_ids)
        type_tokens.append(token_type_ids)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long) # [n, 50]
    all_attention_mask = torch.tensor(attention_masks, dtype=torch.long) # [n, 50]
    all_spans = torch.tensor(spans, dtype=torch.long)  # [n, 4]
    all_type_tokens = torch.tensor(type_tokens, dtype=torch.long) # [n, 50]

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_type_tokens, all_spans)
    trainer = SynonymTrainer(args, test_dataset=dataset)
    return trainer.predict(data_file)

if __name__ == '__main__':
    predict_syn('synonyms_input.txt')