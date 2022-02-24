from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


def configure_optimizers(pl_model):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    model = pl_model.model
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pl_model.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=pl_model.hparams.learning_rate, eps=pl_model.hparams.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=pl_model.hparams.warmup_steps,
        num_training_steps=pl_model.total_steps,
    )
    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]
