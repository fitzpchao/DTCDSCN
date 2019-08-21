import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

class Trainer(object):

    #cuda = torch.cuda.is_available()
    #torch.backends.cudnn.benchmark = True
    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model,device_ids=range(torch.cuda.device_count()))
        self.model.to(device)
        #self.model.load_state_dict(torch.load("checkpoints/sexp6_DA_4/ model_400.pkl")['weight'])
        self.optimizer = optimizer
        self.loss_f = loss_f().cuda()
        self.a_loss_f = torch.nn.BCELoss().cuda()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.writer = SummaryWriter(save_dir)


    def _iteration(self, data_loader, ep ,is_train=True):
        loop_loss = []
        matrixs = np.zeros([2,2],np.float32)
        #accuracy = []
        for img1,img2,target,target_branch1,target_branch2 in tqdm(data_loader):
            #print(data.shape)
            #print(target.shape)
            img1,img2,target,target_branch1,target_branch2 = img1.cuda(),img2.cuda(),target.cuda(),\
                                                             target_branch1.cuda(),target_branch2.cuda()
            output1,output2,output= self.model(img1,img2)
            loss = 0.5*self.loss_f(output,target)  + 0.25 * self.a_loss_f(output1,target_branch1) + 0.25*self.a_loss_f(output2,target_branch2)
            loss_step = loss.data.item()
            print(">>>loss:", loss_step)
            loop_loss.append(loss.data.item() / len(data_loader))
            #accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                output = output.cpu()
                output = output.data.numpy()
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
                target = target.cpu().data.numpy()
                target = np.reshape(target, [-1])
                output = np.reshape(output, [-1])
                target = target.astype(np.int8)
                output = output.astype(np.int8)
                labels = list(set(np.concatenate((target, output), axis=0)))
                if (labels == [0]):
                    matrixs[0, 0] += confusion_matrix(target, output)[0, 0]
                elif (labels == [1]):
                    matrixs[1, 1] += confusion_matrix(target, output)[0, 0]
                else:
                    matrixs += confusion_matrix(target, output)

        if is_train:
            self.writer.add_scalar('train/loss_epoch', sum(loop_loss), ep)
            #self.writer.add_scalar('train/accuracy',sum(accuracy)/len(data_loader.dataset),ep)
        else:
            a, b, c, d = matrixs[0][0], matrixs[0][1], matrixs[1][0], matrixs[1][1]
            print(matrixs)
            accuracy = (a + d) / (a + b + c + d)
            if((d+c)!= 0):
                recall = d / (d + c)
            else:
                recall = 0
            if ((d + b) != 0):
                precision = d / (d + b)
            else:
                precision = 0

            F1 = 2 * d / (a + b + c + d + d - a)
            if ((d + b + c) != 0):
                iou = d / (c + b + d)
            else:
                iou = 0

            self.writer.add_scalar('test/accuracy', accuracy, ep)
            self.writer.add_scalar('test/recall', recall, ep)
            self.writer.add_scalar('test/precision', precision, ep)
            self.writer.add_scalar('test/F1', F1, ep)
            self.writer.add_scalar('test/iou', iou, ep)
            self.writer.add_scalar('test/loss_epoch', sum(loop_loss), ep)
            #self.writer.add_scalar('test/accuracy',sum(accuracy)/len(data_loader.dataset),ep)
        mode = "train" if is_train else "test"
        #print(">>>[{mode}] loss: {loss}/accuracy: {accuracy}".format(mode=mode,loss=sum(loop_loss),accuracy=sum(accuracy)/len(data_loader.dataset)))
        print(">>>[{mode}] loss: {loss}".format(mode=mode,loss=sum(loop_loss)))
        return loop_loss

    def train(self, data_loader,ep):
        self.model.train()
        with torch.enable_grad():
            loss = self._iteration(data_loader,ep)
            #pass

    def test(self, data_loader,ep):
        self.model.eval()
        with torch.no_grad():
            loss = self._iteration(data_loader,ep,is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.train(train_data,ep)
            self.test(test_data,ep)
            if (ep % self.save_freq == 0):
                self.save(ep)

    def save(self, epoch, **kwargs):
        model_out_path = self.save_dir
        state = {"epoch": epoch, "weight": self.model.state_dict()}
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        torch.save(state, model_out_path + '/ model_{epoch}.pkl'.format(epoch=epoch))