import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from utlis import  ChannelCompress, to_edge
from torch.nn import init

class fcModal(nn.Module):
    def __init__(self):
        super(fcModal, self).__init__()
        self.fc_encode = nn.Sequential(
            nn.Linear(settings.CODE_LEN, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(settings.CODE_LEN, 256),
            nn.LeakyReLU(),
            nn.Linear(256, settings.event),
            nn.Softmax(dim=1),
        )
        h_dim = 64
        self.classifier_corre = nn.Sequential(
        nn.Linear(settings.CODE_LEN, h_dim),
        nn.BatchNorm1d(h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.BatchNorm1d(h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, 2)
        )

    def forward(self, x,code_len):

        label_pred = self.classifier_corre(x)
        event_pred = self.domain_classifier(x)
        return label_pred, event_pred

class VIB_I(nn.Module):
    def __init__(self, in_ch=512, z_dim=256, num_class=2):
        super(VIB_I, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)


    def forward(self, v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return  z_given_v

class VIB_T(nn.Module):
    def __init__(self, in_ch=512, z_dim=256, num_class=2):
        super(VIB_T, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
            # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)
        self.fc = nn.Linear(768, 512)

    def forward(self, v):
        v = self.fc(v)
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return  v,z_given_v

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Attention_ast(nn.Module):
    def __init__(self, code_len):
        super(Attention_ast, self).__init__()
        # self.weight = torch.rand(100, 1)
        self.fc = nn.Linear(settings.CODE_LEN, 1)
    def forward(self, x):
        x2 = x
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x2

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.fcModal = fcModal()
        self.attention = Attention_ast(code_len=settings.CODE_LEN)
        self.VIB_I = VIB_I()
        self.VIB_T = VIB_T()
        self.kl_div = torch.nn.KLDivLoss()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, img,txt,label):
        code_I = self.VIB_I(img)
        txt, code_T = self.VIB_T(txt)

        att_A1 = self.attention(code_I)
        att_A2 = self.attention(code_T)
        att_A = torch.cat((att_A1, att_A2), 1)
        att_A = torch.softmax(att_A, dim=1)
        att_A1 = att_A[:, :1]
        att_A2 = att_A[:, 1:]
        F_A = torch.multiply(att_A1, code_I) + torch.multiply(att_A2, code_T)
        label_pred, event_pred = self.fcModal(F_A, settings.BATCH_SIZE)

        F_I = F.normalize(img)
        F_T = F.normalize(txt)
        S_I = F_I.mm(F_I.t())
        B_F_I = S_I
        S_T = F_T.mm(F_T.t())
        B_F_T = S_T
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)
        B_F_a = F.normalize(F_A)
        B_F_A = B_F_a.mm(B_F_a.t())
        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        S_tilde = settings.ALPHA * S_I + (1 - settings.ALPHA) * S_T  # 联合特征矩阵
        S = S_tilde  #
        vsd_loss = self.kl_div(input=self.softmax(img.detach() / 1),
                               target=self.softmax(code_I / 1)) + \
                   self.kl_div(input=self.softmax(txt.detach() / 1),
                               target=self.softmax(code_T / 1))

        loss_fn = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tri_loss = loss_fn(label_pred.to(device), label.long().to(device))
        loss3 = F.mse_loss(B_F_A, S)  # 联合损失
        loss31 = F.mse_loss(BI_BI, settings.K * S_I)  # 模态内损失
        loss32 = F.mse_loss(BT_BT, settings.K * S_T)  # 模态内损失
        loss7 = F.mse_loss(BI_BI, BT_BT)  # 模态间损失
        loss = tri_loss + vsd_loss + loss3 + loss7 + settings.ETA * (loss31 + loss32)

        return loss,label_pred



