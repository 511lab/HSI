import numpy as np
import scipy.io as sio
import spectral
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from operator import truediv
from torch.utils.data import Dataset
from HybridSN import HybridSN
# from utils import utils
# from TrainDS import TrainDS

class_num = 16
windowSize = 25
K = 30        #参考Hybrid-Spectral-Net
rate = 16	

class HybridSN(nn.Module):
  #定义各个层的部分
  def __init__(self):
    super(HybridSN, self).__init__()
    self.S = windowSize
    self.L = K;

    #self.conv_block = nn.Sequential()
    ## convolutional layers
    self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))
    self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))
    self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
    
    #不懂 inputX经过三重3d卷积的大小
    inputX = self.get2Dinput()
    inputConv4 = inputX.shape[1] * inputX.shape[2]
    # conv4 （24*24=576, 19, 19），64个 3x3 的卷积核 ==>（（64, 17, 17）
    self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3))

    #self-attention
    self.sa1 = nn.Conv2d(64, 64//rate, kernel_size=1)
    self.sa2 = nn.Conv2d(64//rate, 64, kernel_size=1)
    
    # 全连接层（256个节点） # 64 * 17 * 17 = 18496
    self.dense1 = nn.Linear(18496, 256) 
    # 全连接层（128个节点）
    self.dense2 = nn.Linear(256, 128)
    # 最终输出层(16个节点)
    self.dense3 = nn.Linear(128, class_num)
    
    #让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
    #但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
    #参考: https://blog.csdn.net/yangfengling1023/article/details/82911306
    #self.drop = nn.Dropout(p = 0.4)
    #改成0.43试试
    self.drop = nn.Dropout(p = 0.43)
    self.soft = nn.Softmax(dim=1)
    pass

  #辅助函数，没怎么懂，求经历过三重卷积后二维的一个大小
  def get2Dinput(self):
    #torch.no_grad(): 做运算，但不计入梯度记录
    with torch.no_grad():
      x = torch.zeros((1, 1, self.L, self.S, self.S))
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
    return x
    pass

  #必须重载的部分，X代表输入
  def forward(self, x):
    #F在上文有定义torch.nn.functional，是已定义好的一组名称
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))

    # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
    out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
    out = F.relu(self.conv4(out))

    # Squeeze 第三维卷成1了
    weight = F.avg_pool2d(out, out.size(2))    #参数为输入，kernel
    #参考: https://blog.csdn.net/qq_21210467/article/details/81415300
    #参考: https://blog.csdn.net/u013066730/article/details/102553073

    # Excitation: sa（压缩到16分之一）--Relu--fc（激到之前维度）--Sigmoid（保证输出为0至1之间）
    weight = F.relu(self.sa1(weight))
    weight = F.sigmoid(self.sa2(weight))
    out = out * weight

    # flatten: 变为 18496 维的向量，
    out = out.view(out.size(0), -1)

    out = F.relu(self.dense1(out))
    out = self.drop(out)
    out = F.relu(self.dense2(out))
    out = self.drop(out)
    out = self.dense3(out)

    #添加此语句后出现LOSS不下降的情况，参考：https://www.e-learn.cn/topic/3733809
    #原因是CrossEntropyLoss()=softmax+负对数损失（已经包含了softmax)。如果多写一次softmax，则结果会发生错误
    #out = self.soft(out)
    #out = F.log_softmax(out)

    return out
    pass

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


# 地物类别
class_num = 16
X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

# 用于测试样本的比例
test_ratio = 0.90

# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 30

print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape) 
print('before transpose: Xtest  shape: ', Xtest.shape) 

# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest  = Xtest.transpose(0, 4, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape) 
print('after transpose: Xtest  shape: ', Xtest.shape) 


""" Training dataset"""
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset  = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128, shuffle=False)

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
net = HybridSN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00037)

# 开始训练
net.train()
total_loss = 0
for epoch in range(200):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))

print('Finished Training')

count = 0
net.eval()
# 模型测试
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    outputs = net(inputs)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test =  outputs
        count = 1
    else:
        y_pred_test = np.concatenate( (y_pred_test, outputs) )

# 生成分类报告
classification = classification_report(ytest, y_pred_test, digits=4)
print(classification)





