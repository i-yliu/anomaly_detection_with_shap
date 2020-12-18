import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import requests
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as pl
import numpy as np
import shap
# import cv2
# load model data
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
model = VGG16()

# load an image
file = "../../data/anomaly/screw/test/scratch_neck/015.png"
# file = "../../data/anomaly/grid/test/bent/001.png"
# file = "../data/rotten_straw.jpg"
img = image.load_img(file, target_size=(224, 224))
img_orig = image.img_to_array(img)

# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=100, compactness=10, sigma=0)

patch = np.zeros((img_orig.shape[0], img_orig.shape[1]))
ind = 0

patch_size = 10

seg = np.zeros(img_orig.shape[:2])
seg_slice = seg.shape[0] // 8

ind = 0
seg_slice = seg.shape[0] // 4
for i in range(seg_slice, seg.shape[0] + 1, seg_slice):
    for j in range(seg_slice, seg.shape[1] + 1, seg_slice):
        seg[i-seg_slice : i, j - seg_slice:j] = ind
        ind += 1

# import pdb; pdb.set_trace()
fig, ax = pl.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(image.array_to_img(img_orig))
# ax[0, 0].set_title("Felzenszwalbs's method")
# import pdb; pdb.set_trace()
ax[0, 1].imshow(mark_boundaries(img, seg.astype('int64')))
ax[0, 1].set_title('SLIC')
# ax[1, 0].imshow(mark_boundaries(img, segments_quick))
# ax[1, 0].set_title('Quickshift')
# ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
# ax[1, 1].set_title('Compact watershed')
pl.show()

# import pdb; pdb.set_trace()
# define a function that depends on a binary mask representing if an image region is hidden
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

def f(z):
    return model.predict(preprocess_input(mask_image(z, seg, img_orig, 255)))
#
# import pdb; pdb.set_trace()
# use Kernel SHAP to explain the network's predictions


explainer = shap.KernelExplainer(f, np.zeros((1,16)))
shap_values = explainer.shap_values(np.ones((1,16)), nsamples='auto') # runs VGG16 1000 times

# get the top predictions from the model
preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))

top_preds = np.argsort(-preds)

# make a color map
from matplotlib.colors import LinearSegmentedColormap
colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))
for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))
cm = LinearSegmentedColormap.from_list("shap", colors)

def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

# plot our explanations
fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12,4))
inds = top_preds[0]
axes[0].imshow(img)
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
for i in range(3):
    m = fill_segmentation(shap_values[inds[i]][0], seg)
    axes[i+1].set_title(feature_names[str(inds[i])][1])
    axes[i+1].imshow(img.convert('LA'), alpha=0.15)
    im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    axes[i+1].axis('off')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
pl.show()
