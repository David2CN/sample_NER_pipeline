import nltk
import xml.etree.ElementTree as ET
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer


def get_tokens_tags(xmls_dir: str, tokenizer=None) -> tuple:

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
                iob_ent = f"B-{entity}"

                # else prefix I for inside an entity chunk
                if vals[0] != sorted(a)[0]:
                    iob_ent = f"I-{entity}"

                ntags.append(iob_ent)
        ner_tags.append(ntags)
    return tokens, ner_tags


def generate_dataset(tokens: list, ner_tags: list) -> dict:
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

    flat_ner_tags = [i for j in ner_tags for i in j]
    result = {
        "dataset": dataset,
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
