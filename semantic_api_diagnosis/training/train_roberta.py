"""RoBERTa multi-label fine-tuning entrypoint."""

from semantic_api_diagnosis.training.train_distilbert import main as train_encoder


def main() -> None:
    train_encoder(default_model_name="roberta-base", model_label="RoBERTa")


if __name__ == "__main__":
    main()
