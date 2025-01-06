import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
 
patch_h = 75
patch_w = 50
feat_dim = 384
 
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
dinov2_vits14 = torch.hub.load('', 'dinov2_vits14',source='local').cuda()
 
features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()
 
img_path = f'demo.png'
img = Image.open(img_path).convert('RGB')
imgs_tensor[0] = transform(img)[:3]
with torch.no_grad():
    features_dict = dinov2_vits14.forward_features(imgs_tensor)
    print("features_dict keys: ",features_dict.keys()) 
    features = features_dict['x_norm_patchtokens']

print("1 features shape: ",features.shape)  

features = features.reshape(4 * patch_h * patch_w, feat_dim).cpu()

print("2 features shape: ",features.shape)

pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)
pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())
 
pca_features_fg = pca_features[:, 0] > 0.3
pca_features_bg = ~pca_features_fg
 
b = np.where(pca_features_bg)

pca.fit(features[pca_features_fg])
pca_features_rem = pca.transform(features[pca_features_fg])
for i in range(3):
    pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
    # transform using mean and std, I personally found this transformation gives a better visualization
    # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_fg] = pca_features_rem
pca_features_rgb[b] = 0

pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
plt.imshow(pca_features_rgb[0][...,::-1])
plt.savefig('features.png')
# plt.show()
# plt.close()
