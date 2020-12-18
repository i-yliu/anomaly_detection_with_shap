import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import numpy as np
import os
import shap
from autoencoder import Autoencoder
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as pl

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform, download = True)
# import pdb; pdb.set_trace()
idx = dataset.targets == 7
# dataset.targets = dataset.targets[idx]
# dataset.train_data = dataset.train_data[idx]

dset_train = torch.utils.data.dataset.Subset(dataset, np.where(idx==1)[0])

# dset_test = torch.utils.data.dataset.Subset(CIFAR100_test, np.where(idx==1)[0])

dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)

model = Autoencoder()
model.load_state_dict(torch.load('./conv_autoencoder.pth'))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

# model.eval()

batch = next(iter(dataloader))
images, _ = batch
####
img = images[100:103]
img = Variable(img)
    # ===================forward=====================
output = model(img, istest=True)
#####
background = images[:100]
test_images = images[100:103]

import pdb; pdb.set_trace()
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)
import pdb; pdb.set_trace()
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
shap.image_plot(shap_numpy, -test_numpy)

#

# for data in dataloader:
#
#     img, _ = data
#     img = Variable(img).cuda()
#         # ===================forward=====================
#     output = model(img)
#     loss = criterion(output, img)
#         # ===================backward====================
#     import pdb; pdb.set_trace()
#     to_explain = img[:16,:]
#     import pdb; pdb.set_trace()
#
#     background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
#
# # explain predictions of the model on four images
#     e = shap.DeepExplainer(model, background)
# # ...or pass tensors directly
# # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
#     shap_values = e.shap_values(x_test[1:5])
#
# # plot the feature attributions
#     shap.image_plot(shap_values, -x_test[1:5])
#
#     e = shap.GradientExplainer((model), to_explain)
#
#     shap_values,indexes = e.shap_values(to_explain, ranked_outputs=2, nsamples=200)
#
#     # get the names for the classes
#     # index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
#
#     # plot the explanations
#     shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
#
#     shap.image_plot(shap_values, to_explain)
#
#     print('epoch [{}/{}], loss:{:.4f}'
#         .format(0, 0, loss.item()))
#     pic = to_img(output.detach().cpu())
#     save_image(pic, './dc_img/image_{}.png'.format('test'))
