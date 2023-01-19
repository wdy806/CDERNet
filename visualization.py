import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os, torch
from torchvision import transforms
from Network import CDERNet
from PIL import Image

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_path', type=str, default='datasets/DEFE/cropped_data/train/', help='My data path.')
   parser.add_argument('--class_num', type=int, default=3, help='Class numbers.')
   parser.add_argument('--embedding_size', type=int, default=256, help='Embedding size.')
   parser.add_argument('-c', '--checkpoint', type=str, default='model/epoch7_lambda0.7_acc0.5111.pth', help='Pytorch checkpoint file path')
   parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 0)')
   return parser.parse_args()

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

args = parse_args()

model = CDERNet(embedding_size=args.embedding_size, class_num=args.class_num, pretrained=None)
if args.checkpoint:
   print("Loading pretrained weights...", args.checkpoint)
   checkpoint = torch.load(args.checkpoint)
   model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model = model.cuda()
model.eval()

targets = []
ebd = np.empty(shape=(0, args.embedding_size), dtype=float)
pic_path = args.data_path
pic_list = os.listdir(pic_path)
with torch.no_grad():
   for pic in pic_list:
      target = int(pic[0])
      targets.append(target)
      image = Image.open(pic_path + pic)
      img_tensor = transform(image)
      img_tensor = img_tensor.cuda()
      img_tensor = torch.unsqueeze(img_tensor, 0)
      embedding, output = model(img_tensor)
      embedding = embedding.cpu()
      ebd_array = embedding.numpy()
      ebd = np.concatenate((ebd, ebd_array))
tar = np.array(targets)
print(tar.shape)
print(ebd.shape)

model = TSNE(n_components=2, verbose=1, random_state=42, perplexity=20)
result = model.fit_transform(ebd)
X = result[:, 0]
Y = result[:, 1]

X = (X - X.min()) / (X.max() - X.min())
Y = (Y - Y.min()) / (Y.max() - Y.min())
color_list = ['deepskyblue', 'chartreuse', 'red']
plt.scatter(X[tar == 0], Y[tar == 0], c=color_list[0], label='neutral', s=5)
plt.scatter(X[tar == 1], Y[tar == 1], c=color_list[1], label='happy', s=5)
plt.scatter(X[tar == 2], Y[tar == 2], c=color_list[2], label='angry', s=5)
plt.legend()
plt.title('t-SNE_2d')
plt.savefig('PCA_2d.png')
plt.show()


model = TSNE(n_components=3, random_state=42, perplexity=15)
result = model.fit_transform(ebd)
X = result[:, 0]
Y = result[:, 1]
Z = result[:, 2]

X = (X - X.min()) / (X.max() - X.min())
Y = (Y - Y.min()) / (Y.max() - Y.min())
Z = (Z - Z.min()) / (Z.max() - Z.min())
color_list = ['deepskyblue', 'chartreuse', 'red']
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
ax.scatter(X[tar == 0], Y[tar == 0], Z[tar == 0], c=color_list[0], label='neutral', s=15)
ax.scatter(X[tar == 1], Y[tar == 1], Z[tar == 1], c=color_list[1], label='happy', s=15)
ax.scatter(X[tar == 2], Y[tar == 2], Z[tar == 2], c=color_list[2], label='angry', s=15)
plt.legend()
plt.title('t-SNE_3d')
plt.savefig('PCA_3d.png')
plt.show()
