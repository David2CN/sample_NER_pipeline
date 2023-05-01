import nltk
import torch
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from random import sample, seed


def preprocess_xmls(xmls_dir: str, tokenizer=None) -> tuple:
    xml_paths = [str(i) for i in Path(xmls_dir).glob("*.xml")]
    
    if not tokenizer:
        tokenizer = nltk.tokenize.NLTKWordTokenizer()
    
    tokens = []
    ner_tags = []
    for xml_file in xml_paths:
        data = ET.parse(xml_file)
        text = data.find("TEXT").text
        
        # words
        words = tokenizer.tokenize(text)
        tokens.append(words)

        # tags
        spans = list(tokenizer.span_tokenize(text))
        tags = data.find("TAGS")

        assert len(spans) == len(words), f"length of spans [{len(spans)}] and words [{len(words)}] should be equal!"

        entities_dict = [dict(tag.items()) for tag in tags]
        entities_span = [(int(i["start"]), int(i["end"])) for i in entities_dict]

        ntags = []
        for i in range(len(spans)):
            # check if word span intersects with any entity span
            a = set(range(int(spans[i][0]), int(spans[i][1])))
            intersects = [i for i in entities_span if a.intersection(set(range(i[0], i[1]))) != set()]

            if len(intersects) == 0:
                # no intersection, not an entity
                # IOB tag is O for outside
                ntags.append("O")
            else:
                # intersection, an entity
                vals = sorted(intersects[0])
                
                # start and end of intersection
                start, end = vals[0], vals[-1]
                entity_idx = entities_span.index((start, end))
                
                # entity type
                entity = entities_dict[entity_idx]["TYPE"]
                # print(f"word: {words[i]}\nspan: {a}\nent_span: {vals}\nentity: {entity}\n")
                
                # if word is at the begining, prefix B- to create IOB tag
                iob_entity = f"B-{entity}"

                # else prefix I for inside an entity chunk
                if vals[0] != sorted(a)[0]:
                    iob_entity = f"I-{entity}"

                ntags.append(iob_entity)
        ner_tags.append(ntags)
    return tokens, ner_tags


def generate_dataset(tokens: list, ner_tags: list, val_split: float=0.2, random_state: int=42) -> dict:
    # get all unique tags
    alltypes = []
    for i in ner_tags:
        alltypes.extend(i)
    alltypes = sorted(set(alltypes), reverse=True)

    # create id2tag and tag2idx dictionaries
    idx2tag = {id: tag for id,tag in enumerate(alltypes)}
    tag2idx = dict(zip(idx2tag.values(), idx2tag.keys()))

    # create df
    data_dict = {
        "id": [str(i) for i in range(len(tokens))],
        "tokens": tokens,
        "ner_tags": [[tag2idx[i] for i in j] for j in ner_tags]
    }

    dataset = Dataset.from_dict(mapping=data_dict,)

    # split dataset
    num_rows = dataset.num_rows
    val_size = int(val_split * num_rows)

    seed(random_state)
    val_indices = sample(range(num_rows), val_size)
    train_indices = [i for i in range(num_rows) if i not in val_indices]

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    flat_ner_tags = [i for j in ner_tags for i in j]
    result = {
        "train": train_dataset,
        "val": val_dataset,
        "idx2tag": idx2tag,
        "tag2idx": tag2idx,
        "names": sorted(set(flat_ner_tags))
    }
    return result


def tokenize_and_align_labels(examples, tokenizer=None, model_name: str="bert-base-uncased"):
    if tokenizer is None:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tag_sentence(text:str, model_dir: str):
    # convert our text to a  tokenized sequence
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(model_dir)
    config_dict = config.to_dict()
    idx2tag = config_dict["id2label"]
    tag2idx = config_dict["label2id"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, id2label=idx2tag, label2id=tag2idx)

    # tokenize inputs
    inputs = tokenizer(text, truncation=True, return_tensors="pt").to(device)
    # get outputs
    outputs = model(**inputs)
    # convert to probabilities with softmax
    probs = outputs[0][0].softmax(1)
    # get the tags with the highest probability
    word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), idx2tag[tagid.item()]) 
                  for i, tagid in enumerate (probs.argmax(axis=1))]

    return pd.DataFrame(word_tags, columns=['word', 'tag'])



