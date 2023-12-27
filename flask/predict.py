import sys
import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from main import predict_syn
from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES

import sys

import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
import xml.dom.minidom

"""
intent_pred :
0 SkillInstructioncon
1 SkillConfirmation
2 Answer_True
3 Answer_False
4 UNK

intent_for_tree :
1 Learn Skill
2 Call Skill
3 Completion
4 Confirm Synonym
5 Positive Answer
6 Negative Answer
100 error 
"""


logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")
    
    model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                                  intent_label_lst=get_intent_labels(args),
                                                                  slot_label_lst=get_slot_labels(args))
    model.to(device)
    model.eval()
    logger.info("***** Model Loaded *****")

    return model

def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:

            words = [word for word in line]
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    args.task = 'myData' 
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file 
    with open(pred_config.output_file, "w", encoding="utf-8") as f, open('PostProcess.txt','a',encoding="utf-8") as f1, \
        open('question_waiting.txt', 'r+', encoding="utf-8") as fq:
        question_waiting = fq.readlines()
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            skill = ''
            param_type = ''
            param = ''
            param_list = []
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
                    if pred == 'B-skill' or pred == 'I-skill': 
                        skill += word
                    else:
                        if pred.startswith('B'):
                            param_list.append(param_type)
                            param_list.append(param)
                            param = ''
                            param_type = pred[2:]
                            param += word
                        elif pred.startswith('I'):
                            param += word
                        else:
                            pass

            param_list.append(param_type)
            param_list.append(param)

            # Fix a speech recognition error case
            if skill == "开启泵" or skill == "开起泵":
                skill = "开气泵"
            if skill == "关起泵":
                skill = "关气泵"

            # Whether the skill with the intent "Skill Instruction" is a learned skill existing in the library
            if intent_pred == 0: # Skill Instruction
                path = r'~/BTScpp/BehaviorTree.CPP/build/examples/myTree.xml'
                tree = ET.parse(path)
                root = tree.getroot()
                intent_for_tree = 1
                if skill == "移动":
                    intent_for_tree = 2
                    if "des" in param_list:
                        skill = "移动#1"
                    elif "dir" in param_list:
                        skill = "移动#2"
                    else:
                        intent_for_tree = 100 # 特殊情况
                        
                else:
                    for BehaviorTree in root.findall('BehaviorTree'):
                        if BehaviorTree.attrib['ID'] == skill:
                            intent_for_tree = 2
                # 如果该技能不在已有子树里，进近义词模型 If the skill is not existing in the library, use the synonymous skill module
                if intent_for_tree == 1: 
                    whole_sentence = ''.join(words).strip()
                    span_left = whole_sentence.find(skill)
                    span_right_1 = span_left + len(skill) - 1
                    f2 = open('synonyms_input.txt', 'w', encoding="utf-8")
                    for BehaviorTree in root.findall('BehaviorTree'):
                        if BehaviorTree.attrib['ID'] == 'BehaviorTree':
                            continue
                        temp_skill = BehaviorTree.attrib['ID']
                        replace_sentence = whole_sentence.replace(skill, temp_skill)
                        span_right_2 = span_left+ len(temp_skill) - 1
                        create_synonym = "{{\"text1\": \"{}\", \"span1\": [{}, {}], \"text2\": \"{}\", \"span2\": [{}, {}]}}"\
                            .format(whole_sentence, span_left, span_right_1, replace_sentence, span_left, span_right_2)
                        f2.write(create_synonym + '\n')
                    f2.close()

                    score_list = predict_syn('synonyms_input.txt') # synonymous skill model
                    best_score = 0
                    for item in score_list:
                        flag, score, s_name = item.split(' ')
                        print(flag, score, s_name)
                        if int(flag) == 1 and float(score) > best_score:
                            intent_for_tree = 4 
                            best_score = float(score)
                            candidate_name = s_name
                            
                    if intent_for_tree == 4:
                        candidate_sentence = whole_sentence.replace(skill, candidate_name)
                        fq.seek(0)
                        fq.write(candidate_sentence+"#"+candidate_name)
                        f1.write("{} 检测到同义词：{}和{}。您是指\"{}\"吗？\n"\
                            .format(intent_for_tree, skill, candidate_name, candidate_sentence)) # Synonymous skills detected: xx and {candidate}. Do you mean...?
                        temp_param = open('synonyms_param.txt', 'w', encoding="utf-8")
                        param_list_str = ",".join(param_list)
                        temp_param.write("{}#{}".format(skill, param_list_str))  
                        temp_param.close()                                                                       
                    

            if intent_pred == 1:
                intent_for_tree = 3 

            if intent_pred == 2:  
                intent_for_tree = 5  
                if len(question_waiting) == 0:
                    f1.write("{} {}\n".format(intent_for_tree, "对不起，我没有明白"))  # Sorry, I don't understand.
                else:
                    write_file = open('input.txt', 'w', encoding="utf-8")
                    write_file.write(question_waiting[0].split("#")[0])
                    f1.write("{} {}\n".format(intent_for_tree, question_waiting[0].split("#")[1]))
                    fq.seek(0)
                    fq.truncate()
            
            if intent_pred == 3:
                intent_for_tree = 6
                if len(question_waiting) == 0:
                    f1.write("{} {}\n".format(intent_for_tree, "对不起，我没有明白")) # Sorry, I don't understand.
                else:
                    read_param = open('synonyms_param.txt', 'r', encoding="utf-8").readlines()
                    h_skill, h_param = read_param[-1].split('#')
                    h_param_list = []
                    for i in h_param.split(','):
                        h_param_list.append(i)
                    f1.write("{} {}".format(1, h_skill))
                    for item in h_param_list:
                        if item != '':
                            f1.write(" "+item)
                    f1.write("\n")
                    fq.seek(0)
                    fq.truncate()
            
            if intent_for_tree == 1 or intent_for_tree == 2 or intent_for_tree == 3 or intent_for_tree == 100:  
                f1.write("{} {}".format(intent_for_tree, skill))
                for item in param_list:
                    if item != '':
                        f1.write(" "+item)
                f1.write("\n")

            f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")



def main():
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="input.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="output.txt", type=str, help="Output file for prediction")
    print(os.getcwd())
    parser.add_argument("--model_dir", default="./my_model", type=str, help="Path to save, load model")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", default="True", action="store_true", help="Avoid using CUDA when available")
    pred_config = parser.parse_args()
    predict(pred_config)
    

if __name__ == "__main__":
    main()