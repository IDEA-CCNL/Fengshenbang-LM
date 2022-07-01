# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jsonlines
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, BertTokenizer
from train_func import CustomDataset, CustomDataModule, CustomModel
import argparse
import os
import gpustat

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "--model_path", default="./weights/Erlangshen-MegatronBert-1.3B-Similarity", type=str, required=False)
    my_parser.add_argument(
        "--model_name", default="IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity", type=str, required=False)
    my_parser.add_argument("--max_seq_length", default=64, type=int, required=False)
    my_parser.add_argument("--batch_size", default=32, type=int, required=False)
    my_parser.add_argument("--val_batch_size", default=64, type=int, required=False)
    my_parser.add_argument("--num_epochs", default=7, type=int, required=False)
    my_parser.add_argument("--learning_rate", default=4e-5, type=float, required=False)
    my_parser.add_argument("--warmup_proportion", default=0.2, type=int, required=False)
    my_parser.add_argument("--warmup_step", default=2, type=int, required=False)
    my_parser.add_argument("--num_labels", default=3, type=int, required=False)
    my_parser.add_argument("--cate_performance", default=False, type=bool, required=False)
    my_parser.add_argument("--use_original_pooler", default=True, type=bool, required=False)
    my_parser.add_argument("--model_output_path", default='./pl_model', type=str, required=False)
    my_parser.add_argument("--mode", type=str, choices=['Train', 'Test'], required=True)
    my_parser.add_argument("--predict_model_path", default='./pl_model/', type=str, required=False)
    my_parser.add_argument("--test_output_path", default='./submissions', type=str, required=False)
    my_parser.add_argument("--optimizer", default='AdamW', type=str, required=False)  # ['Adam', 'AdamW']
    # ['StepLR', 'CosineWarmup', 'CosineAnnealingLR']
    my_parser.add_argument("--scheduler", default='CosineWarmup', type=str, required=False)
    my_parser.add_argument("--loss_function", default='LSCE_correction', type=str,
                           required=False)  # ['CE', 'Focal', 'LSCE_correction']

    args = my_parser.parse_args()

    print(args)
    gpustat.print_gpustat()

    if 'Erlangshen' in args.model_name:
        tokenizer = BertTokenizer.from_pretrained(args.model_name, cache_dir=args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.model_path)

    seed = 1919
    pl.seed_everything(seed)

    dm = CustomDataModule(
        args=args,
        tokenizer=tokenizer,
    )

    metric_index = 2
    checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor=['val_loss', 'val_acc', 'val_f1'][metric_index],
        mode=['min', 'max', 'max'][metric_index]
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                          name='lightning_logs/' + args.model_name.split('/')[-1]),

    trainer = pl.Trainer(
        progress_bar_refresh_rate=50,
        logger=logger,
        gpus=-1 if torch.cuda.is_available() else None,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        max_epochs=args.num_epochs,
        # accelerator='ddp',
        # plugins='ddp_sharded',
    )

    if args.mode == 'Train':
        print('Only Train')
        model = CustomModel(
            args=args,
        )
        trainer.fit(model, dm)

    # Predict test, save results to json
    if args.mode == 'Test':
        print('Only Test')
        test_loader = torch.utils.data.DataLoader(
            CustomDataset('test.json', tokenizer, args.max_seq_length, 'test'),
            batch_size=args.val_batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        model = CustomModel(args=args).load_from_checkpoint(args.predict_model_path, args=args)

        predict_results = trainer.predict(model, test_loader, return_predictions=True)

        path = os.path.join(
            args.test_output_path,
            args.model_name.split('/')[-1].replace('-', '_'))
        file_path = os.path.join(path, 'qbqtc_predict.json')

        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(file_path):
            print('Json文件已存在, 将用本次结果替换')

        with jsonlines.open(file_path, 'w') as jsonf:
            for predict_res in predict_results:
                for i, p in zip(predict_res['id'], predict_res['logits']):
                    jsonf.write({"id": i, "label": str(p)})
        print('Json saved:', file_path)
