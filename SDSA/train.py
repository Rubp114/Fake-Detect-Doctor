import os.path as osp
from torch.utils.data import DataLoader
from models import *
from dataset import *
from triple_loss import *
from sklearn.metrics import accuracy_score,f1_score,classification_report
import numpy as np
import time

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class Session:
    def __init__(self):
        self.logger = settings.logger
        torch.cuda.set_device(settings.GPU_ID)
        print(torch.cuda.is_available())

        # Weibo数据
        if(settings.DATASET== 'Weibo'):
            train_images = np.load(
                r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_clip_train_image.npy",
                allow_pickle=True)
            train_tags = np.load(r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_train_text.npy",
                                 allow_pickle=True)
            train_labels = np.load(r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_train_label.npy",
                                   allow_pickle=True)
            train_event = np.load(r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_train_event.npy",
                                  allow_pickle=True)
            test_images = np.load(
                r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_clip_test_image.npy",
                allow_pickle=True)
            test_tags = np.load(r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_test_text.npy",
                                allow_pickle=True)
            test_labels = np.load(r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_test_label.npy",
                                  allow_pickle=True)
            test_event = np.load(r"D:\BaiduNetdiskDownload\SDSA\weibo_clip\weibo_test_event.npy",
                                 allow_pickle=True)
        if (settings.DATASET == 'Fakeddit'):
            # Fakedict数据集
            train_images = np.load(r"F:/A多模态项目/fakeddit_balance1/train_balance_clip_image.npy",allow_pickle=True)
            train_tags = np.load(r"F:/A多模态项目/fakeddit_balance1/train_balancetext.npy",allow_pickle=True)
            train_labels = np.load(r"F:/A多模态项目/fakeddit_balance1/train_balancelabel.npy",allow_pickle=True)
            train_event = np.load(r"F:/A多模态项目/fakeddit_balance1/train_balancenum.npy",allow_pickle=True)
            test_images = np.load(r"F:/A多模态项目/fakeddit_balance1/test_balance_clip_image.npy", allow_pickle=True)
            test_tags = np.load(r"F:/A多模态项目/fakeddit_balance1/test_balancetext.npy", allow_pickle=True)
            test_labels = np.load(r"F:/A多模态项目/fakeddit_balance1/test_balancelabel.npy", allow_pickle=True)
            test_event = np.load(r"F:/A多模态项目/fakeddit_balance1/test_balancenum.npy", allow_pickle=True)


        # Data Loader (Input Pipeline)
        self.train_dataset = Dataset1(train_images, train_tags, train_labels,  train_event)
        self.test_dataset = Dataset1(test_images, test_tags, test_labels, test_event)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=settings.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=settings.NUM_WORKERS)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=settings.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=settings.NUM_WORKERS)

        self.model1 = model()
        self.opt_model = torch.optim.SGD(self.model1.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.best = 0
    def train(self, epoch):

        self.model1.cuda().train()
        self.logger.info('Epoch [%d/%d]' % (
            epoch + 1, settings.NUM_EPOCH))

        for idx, (img, txt, events, labels) in enumerate(self.train_loader):
            txt = torch.FloatTensor(txt.numpy()).cuda()
            """图像为Tensor类型"""
            img = torch.FloatTensor(img).cuda()

            self.opt_model.zero_grad()
            loss,pred = self.model1(img,txt,labels)
            loss.backward()
            self.opt_model.step()

            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss.item())
                )


    def eval(self,step,  last=False):
        self.model1.eval().cuda()
        re_L = list([])
        test_label_pred1 = list([])
        t0 = time.perf_counter()
        for idx, (img, txt, event,labels) in enumerate(self.test_loader):
            txt_feature = txt
            txt = torch.FloatTensor(txt_feature.numpy()).cuda()
            img = torch.FloatTensor(img.numpy()).cuda()
            loss,test_label_pred = self.model1(img,txt,labels)
            pre_label_detection = test_label_pred.argmax(1)
            test_label_pred1.extend(pre_label_detection.cpu().data.numpy())
            re_L.extend(labels.cpu().data.numpy())

        re_L = np.array(re_L)
        test_label_pred = np.array(test_label_pred1)
        test_accuracy = accuracy_score(re_L, test_label_pred)
        classreport = classification_report(re_L, test_label_pred, digits=3)
        print(test_accuracy)
        self.logger.info('ACC %.4f' % test_accuracy )
        t1 = time.perf_counter()  # 放在测试过程前后

        test_time = 'time:{:.6f}'.format((t1 - t0) / 10000)  # 除以的是测试样本
        print(test_time)
        print(classreport)
        if test_accuracy > self.best:
            self.best = test_accuracy
            self.save_checkpoints(step=step, best=True)
            self.best = test_accuracy
            self.logger.info("#########is best:%.3f #########" % self.best)
        print(self.best)



    def save_checkpoints(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.BATCH_SIZE),
                         best=False):
        if best:
            file_name = '%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'model': self.model1.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.model1.load_state_dict(obj['model'])
        self.logger.info('********** The loaded model has been trained for epochs.*********')

def extract_feature(img,txt):
    img = img
    txt = txt

def main():
    for x in range(1):
        sess = Session()
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % 1 == 0:
                sess.eval(step=epoch + 1)

        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints()
        sess.eval(step=settings.BATCH_SIZE)

if __name__ == '__main__':
    main()
