import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = self.x.size(0)
 
    def __getitem__(self, index):
        return self.x[index], self.y[index]
 
    def __len__(self):
        return self.len


class MyDataset_new(Dataset):
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s
        self.len = self.x.size(0)
 
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.s[index]
 
    def __len__(self):
        return self.len


class CLS_Net(torch.nn.Module):

    def __init__(self, cls_num, z_dim, cls_batch_size):
        super(CLS_Net, self).__init__()

        mini_dim = 256 #256

        out_input_num = mini_dim

        base_dim = 64 #256 #64

        self.cls_batch_size = cls_batch_size
        self.jie = 1

        self.fc1 = nn.Linear(z_dim, mini_dim)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(out_input_num, base_dim)
        self.fc2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(base_dim, cls_num)
        self.out.weight.data.normal_(0, 0.1)

    def self_dis(self, a):
        max_dim = self.cls_batch_size
        jie = self.jie

        all_tag = False
        for j in range(a.shape[0]):
            col_tag = False
            for i in range(a.shape[0]):
                tmp = F.pairwise_distance(a[j,:], a[i,:] , p = jie).view(-1,1)
                if col_tag == False:
                    col_dis = tmp
                    col_tag = True
                else:
                    col_dis = torch.cat((col_dis, tmp), dim = 0)
            if all_tag == False:
                all_dis = col_dis
                all_tag = True
            else:
                all_dis = torch.cat((all_dis, col_dis), dim = 1)
        '''
        print(all_dis.shape)
        if all_dis.shape[1] < max_dim:
            all_dis = torch.cat((all_dis, all_dis[:,:(max_dim - all_dis.shape[1])]), dim = 1)
        print(all_dis.shape)
        '''
        return all_dis

    def forward(self, x):

        x = self.fc1(x)
        x1 = F.relu(x)

        x2 = self.fc2(x1)
        x2 = torch.nn.Dropout(0.1)(x2) #0.3
        x2 = F.relu(x2)

        y = self.out(x2)

        return y, x1


class Gen_Net(torch.nn.Module):

    def __init__(self,input_x2_dim, output_dim):
        super(Gen_Net, self).__init__()

        self.x2_input = nn.Linear(input_x2_dim , 60)
        self.x2_input.weight.data.normal_(0, 0.1)

        self.fc1 = nn.Linear(60, 128)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(128, output_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self,x2):
        x2 = self.x2_input(x2)

        x = x2
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)
        y = self.out(x)

        return y


class gans_process():

    def __init__(self, config):
        
        #base pare
        self.device = config.device
        self.cls_num = config.cls_num
        self.x2_dim =  config.noise_dim
        self.z_dim = config.z_dim

        self.cls_lr = config.cls_lr
        self.gen_lr = config.gen_lr 
        self.cls_epoches = config.cls_epoches
        self.gen_epoches = config.gen_epoches
        self.mse_weight = 1.0

        self.cls_batch_size = config.cls_batch_size
        self.gen_batch_size = config.gen_batch_size
        self.eval_batch_size = config.cls_batch_size
        self.gen_batch_size = self.cls_batch_size

        #optimer and net
        self.cls_net = CLS_Net(self.cls_num, self.z_dim, self.cls_batch_size).to(self.device)
        self.cls_optimizer = torch.optim.SGD(self.cls_net.parameters(), 
                                             lr = self.cls_lr , weight_decay= 1e-5)
        # gen net
        self.gen_net = Gen_Net(self.x2_dim, self.z_dim).to(self.device)

        self.gen_optimizer = torch.optim.SGD(self.gen_net.parameters(), 
                                             lr = self.gen_lr , weight_decay= 0.01)

        #base loss
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_mse = torch.nn.MSELoss()

    def freeze_cls(self):
        for param in self.cls_net.parameters():
            param.requires_grad = False

    def unfreeze_cls(self):
        for param in self.cls_net.parameters():
            param.requires_grad = True

    def freeze_gen(self):
        for param in self.gen_net.parameters():
            param.requires_grad = False

    def unfreeze_gen(self):
        for param in self.gen_net.parameters():
            param.requires_grad = True
    
    def labels2genx(self, sample_num):
        x = torch.rand(sample_num, self.x2_dim)
        return x.to(self.device)

    def pad_batch(self, x):
        if int(x.shape[0] % self.cls_batch_size) == 0:
            return x
        pad_len = self.cls_batch_size  - ( x.shape[0] % self.cls_batch_size)
        x = torch.cat((x, x[:pad_len]), dim = 0)
        return x

    def ready_cls(self, sent_output,perm=None):
        sample_num = len(sent_output)
        #---------------make fake z---------------
        sent_output = sent_output.to(self.device)
        sent_noise = torch.tensor(self.gen_test(sample_num)).to(self.device)

        #--------------handle datas---------------
        x = torch.cat((sent_output, sent_noise), dim = 0 )
        if perm is None:
            perm = torch.randperm(len(x))
        x = x[perm]
        #add y - only one label per time
        multi_label_num = 1
        multi_output_y = torch.tensor([0]*sample_num).unsqueeze(1)
        multi_noise_y = torch.zeros([sent_noise.size(0),1], dtype = torch.int)
        multi_noise_y = multi_noise_y + multi_label_num

        y = torch.cat((multi_output_y, multi_noise_y), dim = 0).to(self.device)
        y = y[perm]
        # x_train = x [:self.train_len]
        # y_train = y [:self.train_len]
        # x_test =  x [self.train_len:]
        # y_test = y [self.train_len:]
        
        return x,y,None,None,perm

    def ready_fake(self, sent_output, inputs_labels, inputs_indexs, label2id, perm = None):

        #---------------make fake z---------------
        sent_output = sent_output.to(self.device)
        sent_noise = torch.tensor(self.gen_test(inputs_labels, inputs_indexs)).to(self.device)

        #--------------handle datas---------------
        x = sent_noise
        y = torch.tensor(inputs_labels).unsqueeze(1)
        if perm is None:
            perm = torch.randperm(len(x))
        x = x[perm]
        y = y[perm]

        return x,y,perm

    def ready_gen(self, sent_output):
        #, inputs_labels, inputs_indexs
        sent_num = len(sent_output)
        sent_output = sent_output.to(self.device)
        x2 = self.labels2genx(sent_num)
        y = torch.tensor([0]*sent_num).unsqueeze(1).to(self.device)

        return x2, y, sent_output

    def cls_train(self, x, y, if_oneHot = True):
        
        #init
        self.cls_net.train()
        self.gen_net.eval()

        self.unfreeze_cls()
        self.freeze_gen()

        x = x.to(self.device)
        y = y.to(self.device)

        #if oneHot
        if if_oneHot:
            y = torch.zeros(y.size(0), self.cls_num).to(self.device).scatter_(1, y.long(), 1)
        #make dataset
        mydataset = MyDataset(x, y)
        train_loader = DataLoader(dataset=mydataset, 
                                  batch_size=self.cls_batch_size, shuffle=True)

        #training
        for epoch in range(self.cls_epoches):
            losses = []
            accuracy = []
            for step, (batch_x, batch_y) in enumerate(train_loader):
                self.cls_optimizer.zero_grad()

                out, _ = self.cls_net(batch_x)
                loss = self.loss_func(out, batch_y)

                #One-side label smoothing -not used
                #location 0 real, location 1 fake
                batch_y = batch_y * torch.tensor([0.9, 1.0]).to(self.device)

                loss.backward()       
                self.cls_optimizer.step()
                #tqdm
                _, predictions = out.max(1)
                predictions = predictions.cpu().numpy().tolist()
                _,real_y = batch_y.max(1)
                real_y = real_y.cpu().numpy().tolist()

                num_correct = np.sum([int(x==y) for x,y in zip(predictions, real_y)])
                running_train_acc = float(num_correct) / float(batch_x.shape[0])
                losses.append(loss)
                accuracy.append(running_train_acc)


        return self.cls_net
    
    def cls_eval(self, x, y, if_oneHot = True):

        #init
        self.cls_net.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        #if oneHot
        if if_oneHot:
            y = torch.zeros(y.size(0), self.cls_num).to(self.device).scatter_(1, y.long(), 1)
        #make dataset
        mydataset = MyDataset(x, y)
        train_loader = DataLoader(dataset=mydataset, 
                                  batch_size=self.eval_batch_size, shuffle=False)

        losses = []
        accuracy = []
        #evaling
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out,_ = self.cls_net(batch_x)
            loss = self.loss_func(out, batch_y)

            #tqdm
            _, predictions = out.max(1)
            predictions = predictions.cpu().numpy().tolist()
            _,real_y = batch_y.max(1)
            real_y = real_y.cpu().numpy().tolist()

            num_correct = np.sum([int(x==y) for x,y in zip(predictions, real_y)])
            running_train_acc = float(num_correct) / float(batch_x.shape[0])
            accuracy.append(running_train_acc)


        mean_acc = np.mean(accuracy)
        return mean_acc

    def cls_real_eval(self, x, y, if_oneHot = True):

        #init
        self.cls_net.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        #if oneHot
        if if_oneHot:
            y = torch.zeros(y.size(0), self.cls_num).to(self.device).scatter_(1, y.long(), 1)
        #make dataset
        mydataset = MyDataset(x, y)
        train_loader = DataLoader(dataset=mydataset, 
                                  batch_size=self.eval_batch_size, shuffle=False)

        rs = 0
        alls = 0

        #evaling
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out, _ = self.cls_net(batch_x)
            loss = self.loss_func(out, batch_y)

            #tqdm
            _, predictions = out.max(1)
            predictions = predictions.cpu().numpy().tolist()
            _,real_y = batch_y.max(1)
            real_y = real_y.cpu().numpy().tolist()

            right_num = np.sum([int( x==y and int(y) != int(self.cls_num-1) ) for x,y in zip(predictions, real_y)])
            all_num = np.sum([int(int(y) != int(self.cls_num-1) ) for x,y in zip(predictions, real_y)])

            rs = rs + right_num
            alls = alls + all_num


        return rs/alls

    def cls_test(self, x, if_oneHot = True):

        #init
        self.cls_net.eval()
        x = x.to(self.device)
        y = torch.zeros([x.size(0),1], dtype = torch.float).to(self.device)

        #if oneHot
        if if_oneHot:
            y = torch.zeros(y.size(0), self.cls_num).to(self.device).scatter_(1, y.long(), 1)
        #make dataset
        mydataset = MyDataset(x, y)
        train_loader = DataLoader(dataset=mydataset, 
                                  batch_size=self.eval_batch_size, shuffle=False)

        preds = []
        #testing
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out, _ = self.cls_net(batch_x)
            loss = self.loss_func(out, batch_y)

            #tqdm
            _, predictions = out.max(1)
            predictions = predictions.cpu().numpy().tolist()
            preds.extend(predictions)

        return preds

    def gen_train(self, x2, y, s, times):

        #init
        self.cls_net.eval()
        self.gen_net.train()
        
        self.freeze_cls()
        self.unfreeze_gen()

        #y is gen + cls
        y = torch.zeros(y.size(0), self.cls_num).to(self.device).scatter_(1, y.long(), 1)
        
        #make dataset
        mydataset = MyDataset_new(x2, y, s)
        train_loader = DataLoader(dataset=mydataset, 
                                  batch_size=self.gen_batch_size, shuffle=True)

        #training
        for epoch in range(self.gen_epoches):
            losses = []
            accuracy = []
            for step, (batch_x2, batch_y, batch_s) in enumerate(train_loader):
                
                # no zero_grad = make batch_size
                if step % 6 == 5: #23
                    self.gen_optimizer.zero_grad()

                out = self.gen_net(batch_x2)

                #fearture matching
                out, hds = self.cls_net(out)
                out2, hds2 = self.cls_net(batch_s.float())
                loss = self.loss_mse(hds, hds2)
                loss = loss * pow(0.9, times)
                loss.backward()
                self.gen_optimizer.step()

                #tqdm
                _, predictions = out.max(1)
                predictions = predictions.cpu().numpy().tolist()
                _, real_y = batch_y.max(1)
                real_y = real_y.cpu().numpy().tolist()

                num_correct = np.sum([int(x==y) for x,y in zip(predictions, real_y)])
                running_train_acc = float(num_correct) / float(batch_x2.shape[0])
                losses.append(loss)
                accuracy.append(running_train_acc)

        return self.gen_net

    def gen_test(self, sample_num):

        #init
        self.gen_net.eval()
        x2 = self.labels2genx(sample_num)
        #x2: len(inputs_labels) * 80
        y = torch.zeros([sample_num,1], dtype = torch.float)
        y = torch.zeros(sample_num, self.z_dim).scatter_(1, y.long(), 1)
        y = y.to(self.device)
        s = torch.ones((sample_num, self.z_dim)).to(self.device)

        #make dataset
        mydataset = MyDataset_new(x2, y, s)
        train_loader = DataLoader(dataset=mydataset, 
                                  batch_size=self.eval_batch_size, shuffle=False)

        preds = []
        #testing
        for step, (batch_x2, batch_y, batch_s) in enumerate(train_loader):

            out = self.gen_net(batch_x2)
                
            loss = self.loss_mse(out.double(), batch_s.double())

            predictions = out.cpu().detach().numpy().tolist()
            preds.extend(predictions)

        return preds


if __name__ == '__main__':

    pass

