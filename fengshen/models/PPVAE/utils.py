from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get data
        d = self.data[index]
        return d

class EarlyStopping():
    def __init__(self, tolerance=10, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, min_loss):
        if (train_loss-min_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

# def gen_text_from_center(args,plugin_vae, vae_model, decoder_tokenizer,label,epoch,pos):
#     gen_text = []
#     latent_z = gen_latent_center(plugin_vae,pos).to(args.device).repeat((1,1))
#     print("latent_z",latent_z.shape)
#     text_analogy = text_from_latent_code_batch(latent_z, vae_model, args, decoder_tokenizer)
#     print("label",label)
#     print(text_analogy)
#     gen_text.extend([(label,y,epoch) for y in  text_analogy])
#     text2out(gen_text, '/cognitive_comp/liangyuxin/projects/cond_vae/outputs/test.json')