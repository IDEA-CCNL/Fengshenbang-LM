from pytorch_lightning.callbacks import ModelCheckpoint


class UniversalCheckpoint():
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('universal checkpoint callback')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--save_ckpt_path', default='./ckpt/', type=str)
        parser.add_argument('--load_ckpt_path', default='./ckpt/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=False)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=None, type=float)
        parser.add_argument('--save_weights_only', action='store_true', default=False)
        parser.add_argument('--every_n_epochs', default=None, type=int)
        parser.add_argument('--save_on_train_epoch_end', action='store_true', default=None)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.save_ckpt_path,
                                         filename=args.filename,
                                         save_last=args.save_last,
                                         every_n_epochs=args.every_n_epochs,
                                         save_on_train_epoch_end=args.save_on_train_epoch_end)
