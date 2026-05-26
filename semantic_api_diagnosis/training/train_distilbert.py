"""DistilBERT multi-label fine-tuning scaffold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from semantic_api_diagnosis.training.data import build_training_records
from semantic_api_diagnosis.training.labels import ERROR_LABELS
from semantic_api_diagnosis.training.metrics import compute_multilabel_metrics
from semantic_api_diagnosis.training.modeling import build_sequence_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for multi-label API error diagnosis.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="outputs/distilbert_stage6")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dry-run", choices=["true", "false"], default="true")
    args = parser.parse_args()

    train_records = build_training_records(args.train)
    validation_records = build_training_records(args.validation)
    if args.dry_run == "true":
        _dry_run(args, train_records, validation_records)
        return
    _train(args, train_records, validation_records)


def _dry_run(args: argparse.Namespace, train_records: list[dict], validation_records: list[dict]) -> None:
    print("DistilBERT training dry-run")
    print(f"Model name: {args.model_name}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Number of labels: {len(ERROR_LABELS)}")
    print("Train label counts:")
    for label, count in _label_counts(train_records).items():
        print(f"- {label}: {count}")
    _verify_tokenizer_if_available(args.model_name, args.max_length, train_records[:2])
    print("Dry-run complete: no training was run and no model was saved.")


def _train(args: argparse.Namespace, train_records: list[dict], validation_records: list[dict]) -> None:
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Full training requires optional dependencies. Install with: "
            'python3 -m pip install -e ".[train]"'
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = Dataset.from_list(train_records).map(
        lambda batch: _tokenize_batch(tokenizer, batch, args.max_length),
        batched=True,
    )
    validation_dataset = Dataset.from_list(validation_records).map(
        lambda batch: _tokenize_batch(tokenizer, batch, args.max_length),
        batched=True,
    )
    model = build_sequence_classifier(args.model_name, num_labels=len(ERROR_LABELS))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=_trainer_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "model"))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Training complete. Metrics written to {output_dir / 'metrics.json'}")


def _tokenize_batch(tokenizer, batch: dict, max_length: int) -> dict:
    tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["labels"] = batch["labels"]
    return tokenized


def _trainer_metrics(eval_prediction) -> dict:
    import math

    logits, labels = eval_prediction
    probabilities = [
        [1 / (1 + math.exp(-float(value))) for value in row]
        for row in logits
    ]
    metrics = compute_multilabel_metrics(labels, probabilities, threshold=0.5)
    return {
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "exact_match": metrics["exact_match"],
        "semantic_macro_f1": metrics["semantic_macro_f1"],
        "semantic_micro_f1": metrics["semantic_micro_f1"],
    }


def _label_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        for label, value in zip(ERROR_LABELS, record["labels"]):
            if value:
                counts[label] += 1
    return {label: counts[label] for label in ERROR_LABELS}


def _verify_tokenizer_if_available(model_name: str, max_length: int, sample_records: list[dict]) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print('Tokenizer check skipped: transformers is not installed. Install with python3 -m pip install -e ".[train]".')
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:
        print(f"Tokenizer check skipped: {model_name} is not available locally ({exc}).")
        return
    texts = [record["text"] for record in sample_records]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    print(f"Tokenizer check passed: tokenized {len(texts)} examples with max_length={max_length}.")
    print(f"Tokenized fields: {', '.join(sorted(tokenized.keys()))}")


if __name__ == "__main__":
    main()
