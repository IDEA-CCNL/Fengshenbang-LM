from pytorch_lightning.callbacks import ModelCheckpoint
import os


class UniversalCheckpoint(ModelCheckpoint):
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('universal checkpoint callback')

        parser.add_argument('--monitor', default='step', type=str)
        parser.add_argument('--mode', default='max', type=str)
        parser.add_argument('--save_ckpt_path', default='./ckpt/', type=str)
        parser.add_argument('--load_ckpt_path', default='./ckpt/', type=str)
        parser.add_argument(
            '--filename', default='model-ep{epoch:02d}-st{step:d}', type=str)
        parser.add_argument('--save_last', action='store_true', default=False)
        parser.add_argument('--save_top_k', default=10, type=float)
        parser.add_argument('--every_n_train_steps', default=None, type=float)
        parser.add_argument('--save_weights_only', action='store_true', default=False)
        parser.add_argument('--every_n_epochs', default=None, type=int)
        parser.add_argument('--save_on_train_epoch_end', action='store_true', default=None)

        return parent_args

    def __init__(self, args):
        super().__init__(monitor=args.monitor,
                         save_top_k=args.save_top_k,
                         mode=args.mode,
                         every_n_train_steps=args.every_n_train_steps,
                         save_weights_only=args.save_weights_only,
                         dirpath=args.save_ckpt_path,
                         filename=args.filename,
                         save_last=args.save_last,
                         every_n_epochs=args.every_n_epochs,
                         save_on_train_epoch_end=args.save_on_train_epoch_end)

        # 做兼容，如果目录不存在的话把这个参数去掉，不然会报错
        if args.load_ckpt_path is not None and \
                not os.path.exists(args.load_ckpt_path):
            print('--------warning no checkpoint found--------, remove args')
            args.load_ckpt_path = None
