import argparse
import evaluate 
import numpy as np

from pathlib import Path
from utils import preprocess_xmls, generate_dataset, tokenize_and_align_labels
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments


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

    metric_seqeval = evaluate.load("seqeval")
    results = metric_seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def run(data_dir: str, model_name: str="bert-base-uncased", output_dir: str="logs/", 
        num_epochs: int=3, val_split: float=0.2, random_state: int=42, max_steps: int=1000):

    tokens, tags = preprocess_xmls(data_dir)

    data = generate_dataset(tokens, tags, val_split=val_split, random_state=random_state)
    idx2tag = data["idx2tag"]
    tag2idx = data["tag2idx"]
    train_dataset = data["train"]
    val_dataset = data["val"]
    
    global tag_labels
    tag_labels = data["names"]

    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    # build model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(model_name,  
                                                            id2label=idx2tag, label2id=tag2idx)

    # training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=2e-5,
        per_device_train_batch_size=64,   
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=500, 
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 6)]
    )

    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, default="./data/", help="path to directory containing xml files")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="name of transformer model to load, defaults to 'bert-base-uncased'")
    parser.add_argument("--output-dir", type=str, default="./logs/", help="name of output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="number of epochs to train for")
    parser.add_argument("--val-split", type=float, default=0.2, help="percentage of data to use for validation")
    parser.add_argument("--random-state", type=int, default=42, help="random seed")
    parser.add_argument("--max-steps", type=int, default=1000, help="max steps")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    print(vars(opt))
    run(**vars(opt))


