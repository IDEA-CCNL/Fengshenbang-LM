import argparse
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Config
# from transformers import AutoProcessor, HubertConfig, HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
import torch
from fengshen.data.wav2vec2.wav2vec2ctc_dataset import CTCDataset
from decoder import build_ctcdecoder
from tqdm import tqdm


def get_model(model_path, ckpt):
    config_type = Wav2Vec2Config
    model_type = Wav2Vec2ForCTC

    config = config_type.from_pretrained(model_path)
    # config = HubertConfig.from_pretrained(model_path)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    config.update(
        {
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
        }
    )
    config.vocab_size = len(processor.tokenizer.get_vocab())
    model = model_type.from_pretrained(ckpt, config=config)
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)

    return model, processor


def prepare_dataloader(processor, test_tsv, test_wrd):
    from torch.utils.data import DataLoader
    dataset = CTCDataset(test_tsv, 16000, test_wrd, processor, processor.tokenizer, processor.feature_extractor, max_sample_size=700000, shuffle=False)
    val_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collater,
        pin_memory=True,
    )
    return val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--ckpt")
    parser.add_argument("--tsv")
    parser.add_argument("--wrd")
    parser.add_argument("--target")
    parser.add_argument("--lm_path")
    args = parser.parse_args()
    model, processor = get_model(args.model_path, args.ckpt)
    val_loader = prepare_dataloader(processor, args.tsv, args.wrd)
    model.half()
    device = torch.device("cuda:0")
    model.to(device)
    predict = []
    if args.lm_path:
        vocab = processor.tokenizer.get_vocab()
        id2vocab = {k: i for i, k in vocab.items()}
        labels = []
        for i in range(4472):
            if id2vocab[i] == "<pad>":
                labels.append("")
            elif id2vocab[i] == "<unk>":
                labels.append("\u2047")
            elif id2vocab[i] == "|":
                labels.append(" ")
            else:
                labels.append(id2vocab[i])
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=args.lm_path,
        )

        lm_processor = Wav2Vec2ProcessorWithLM(processor.feature_extractor, processor.tokenizer, decoder)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            for item in batch:
                if torch.is_tensor(batch[item]):
                    batch[item] = batch[item].to(device)
                    if batch[item].dtype == torch.float:
                        batch[item] = batch[item].to(torch.float16)

            if args.lm_path:
                logits = model(**batch).logits.cpu().numpy()
                outputs = lm_processor.batch_decode(logits, beam_width=400)
                text = outputs.text
                text = [t.replace(" ", "") for t in text]
                predict += text
            else:
                output = model(**batch)
                pred_logits = output.logits
                pred_ids = torch.argmax(pred_logits, axis=-1)
                text = processor.tokenizer.batch_decode(pred_ids)
                text = [t.replace(" ", "") for t in text]
                predict += text

    with open(args.target, 'w') as f:
        f.write("\n".join(predict))
        f.write("\n")
