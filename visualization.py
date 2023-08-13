import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import os
import torch
import yaml
from ignite.contrib import metrics
import constants as const
import dataset
import fastflow
import utils
from PIL import Image
from torchvision import transforms
from fuse_main import build_model
from thop import profile
from thop import clever_format

image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

Config = const.DefaultConfig()
backbone_name = 'WResnet50'
class_name = 'cable'
Config.parse({'backbone_name': backbone_name})
Config.parse({'class_name': class_name})



model = build_model(Config)
model = model.cuda()

checkpoint = torch.load(os.path.join(os.path.join(const.CHECKPOINT_DIR, Config.class_name), Config.backbone_name+"_fuseflow.pt"))
model.nf_flows.load_state_dict(checkpoint["model_nf_flows"])
model.norms.load_state_dict(checkpoint["model_norms_state_dict"])

model = model.eval()
image_file = r'D:\IMSN-YHM\dataset\cable\test\cable_swap\001.png'
image = Image.open(image_file).convert('RGB').resize((256, 256))
image_tensor = image_transform(image).unsqueeze(0).cuda()

import time

for i in range(10):
    t1 = time.time()
    output = model(image_tensor)
    t2 = time.time()
    print(t2-t1)
# with torch.no_grad():
#     memory_before = torch.cuda.memory_allocated()
#     output = model(image_tensor)
#     # Get final memory usage
#     memory_after = torch.cuda.memory_allocated()
#     memory_usage = (memory_after - memory_before) / (1024 ** 2)  # in MB
#     print("Memory usage: ", memory_usage, " MB")

with torch.no_grad():
    features = model.feature_extractor(image_tensor)
    x_local, x_global = model.reconstructor.forward_(features)
print(features[0].shape)
print(x_local[0].shape)
print(x_global[0].shape)


plt.figure()
plt.subplot(131)
plt.imshow(features[0][0][2].clone().cpu().detach().numpy())
plt.subplot(132)
plt.imshow(x_local[0][0][2].clone().cpu().detach().numpy())
plt.subplot(133)
plt.imshow(x_global[0][0][2].clone().cpu().detach().numpy())
plt.show()

anomaly_map = output["anomaly_map"].cpu().detach().squeeze(0).squeeze(0)

plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow((1+anomaly_map)**2, cmap='jet')
plt.colorbar()
plt.show()

