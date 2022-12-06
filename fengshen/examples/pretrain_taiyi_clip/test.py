from pytorch_lightning import (
    Trainer,
)
from fengshen.models.model_utils import (
    add_module_args,
)
import argparse
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.examples.pretrain_taiyi_clip.pretrain import (
    TaiyiCLIP,
    Collator,
)
from fengshen.data.fs_datasets import load_dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = TaiyiCLIP.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()
    checkpoint_callback = UniversalCheckpoint(args)
    trainer = Trainer.from_argparse_args(args, callbacks=[
        checkpoint_callback
    ])

    model = TaiyiCLIP(args)
    processor = model.processor
    collate_fn = Collator(processor)
    datasets = load_dataset(args.datasets_name)
    dataloader = DataLoader(datasets[args.test_datasets_field],
                            batch_size=args.test_batchsize, num_workers=2, collate_fn=collate_fn)
    trainer.validate(model, dataloaders=dataloader, ckpt_path=args.load_ckpt_path)
