import torch
import evaluate 
import numpy as np
import pandas as pd
from utils import get_tokens_tags, generate_dataset, tokenize_and_align_labels
from sklearn.dummy import DummyClassifier
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments


xmls_dir = "./data/"
tokens, tags = get_tokens_tags(xmls_dir)
data = generate_dataset(tokens, tags)

idx2tag = data["idx2tag"]
dataset = data["dataset"]
tag_labels = data["names"]
print(dataset)
print(tag_labels)

tokenized_train_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# baseline
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(pd.Series(tokenized_train_dataset['input_ids']).explode(), pd.Series(tokenized_train_dataset['labels']).explode().astype(str))
score = dummy_clf.score(pd.Series(tokenized_train_dataset['input_ids']).explode(), pd.Series(tokenized_train_dataset['labels']).explode().astype(str))
print(score)


# model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag_labels))

# train
metric_seqeval = evaluate.load("seqeval")
example = tokenized_train_dataset[2]

labels = [tag_labels[i] for i in example["ner_tags"]]
res = metric_seqeval.compute(predictions=[labels], references=[labels])
print(res)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [tag_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tag_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric_seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir='./log_results',
    num_train_epochs=50,
    learning_rate=2e-5,
    per_device_train_batch_size=16,   
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    warmup_steps=500, 
    eval_steps=60,
    save_steps=60,
    evaluation_strategy="steps",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 6)]
)

trainer.train()


# inference
def tag_sentence(text:str):
    # convert our text to a  tokenized sequence
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, truncation=True, return_tensors="pt").to(device)

    # get outputs
    outputs = model(**inputs)
    # convert to probabilities with softmax
    probs = outputs[0][0].softmax(1)
    # get the tags with the highest probability
    word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), idx2tag[tagid.item()]) 
                  for i, tagid in enumerate (probs.argmax(axis=1))]

    return pd.DataFrame(word_tags, columns=['word', 'tag'])


text = """Celebrities and tourists from United States are 
flooding into Greece. But a harsh winter isn’t far off"""
text = """February 12, 2106 
Har is a 43 year old 6' 214 pound gentleman who is referred for
consultation by Dr. Harlan Oneil."""

print(tag_sentence(text))

