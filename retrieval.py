import os, sys
import copy

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import torch, math, time, argparse, json
import random, dataset, utils, net
import numpy as np
import pdb

from dataset.Inshop import Inshop_Dataset

from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
from tqdm import *

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

class CUBClassNames():
    """
    Mapping indices to CUB class names
    """
    IDS_TO_NAMES = {
        0: "Brown Pelican",
        1: "White Pelican",
        2: "Western_Wood_Pewee",
        3: "Sayornis",
        4: "American_Pipit",
        5: "Whip_poor_Will",
        6: "Horned_Puffin",
        7: "Common_Raven",
        8: "White_necked_Raven",
        9: "American_Redstart",
        10: "Geococcyx",
        11: "Loggerhead_Shrike",
        12: "Great_Grey_Shrike",
        13: "Baird_Sparrow",
        14: "Black_throated_Sparrow",
        15: "Brewer_Sparrow",
        16: "Chipping_Sparrow",
        17: "Clay_colored_Sparrow",
        18: "House_Sparrow",
        19: "Field_Sparrow",
        20: "Fox_Sparrow",
        21: "Grasshopper_Sparrow",
        22: "Harris_Sparrow",
        23: "Henslow_Sparrow",
        24: "Le_Conte_Sparrow",
        25: "Lincoln_Sparrow"
    }

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'  
    + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 1, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'resnet50', type = str,
    help = 'Model for training'
)
parser.add_argument('--method', default = 'baseline', type = str,
    help = 'Baseline or metrix method'
)
parser.add_argument('--num_classes', default = 100, type = int,
    help = 'Number of classes to be used for tSNE'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--resume', default = '',
    help = 'Path of resuming model'
)
parser.add_argument('--compute_embeddings', action='store_true',
    help = 'Compute or not embeddings'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)
parser.add_argument('--n_components', default = 2, type = int,                                                                                                                                                                                        
help = 'Number of components for tSNE'                                                                                                                                                                                                                 
)
parser.add_argument('--perplexity', default = 10.0, type = float,
help = 'tSNE perplexity parameter'
)
parser.add_argument('--n_iter', default = 7500, type = int, 
help = 'Number of iterations for tSNE'
)
parser.add_argument('--mode', default = 'embeddings', type = str,
help = 'Use embeddings or features for tSNE'
)
parser.add_argument('--seed', default = 1994, type = int, help = 'RNG seed')
args = parser.parse_args()

if args.mode == 'embeddings':
    from resnet import *
else:
    from resnet_features import *

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Root Directory
workdir = '/work2/pa17/rslab/bill/'
data_root = os.path.join(workdir, 'datasets')
    
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
            name = args.dataset,
            root = data_root,
            mode = 'eval',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
            ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )
    
else:
    query_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
    ))
    
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

    gallery_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'gallery',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
    ))
    
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

# Backbone Model
if args.model.find('googlenet')+1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('resnet18')+1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('resnet50')+1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('resnet101')+1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
#model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

# Make two copies of the model
baseline_model = copy.deepcopy(model)
metrix_model = copy.deepcopy(model)

# Load baseline and metrix model
print('Loading baseline model...')
baselinecheck = torch.load('/users/pa17/rslab/bill/code/metrix/new-mixup/CUB/tensor/weights/baseline.t7', map_location=device)
baseline_model.load_state_dict(baselinecheck['model'].state_dict())
baseline_model.to(device)

print('Loading metrix model...')
metrix_model = torch.nn.DataParallel(metrix_model)
metrixcheck = torch.load('/users/pa17/rslab/bill/code/metrix/new-mixup/CUB/tensor/weights/metrix.t7', map_location=device)
metrix_model.load_state_dict(metrixcheck['model'].state_dict())
metrix_model.to(device)

def computeXT(dl_ev, model):
    with torch.no_grad():
        model.eval()
        ds = dl_ev.dataset
        A = [[] for i in range(len(ds[0]))]
        for batch in tqdm(dl_ev):
            for i, J in enumerate(batch):
                if i==0:
                    J = model(J)
                for j in J:
                    A[i].append(J)

    A[0] = A[0][:5904]
    A[1] = A[1][:5904]

    X, T = [torch.stack(A[i]) for i in range(len(A))] 

    X = torch.squeeze(X, 1)
    T = torch.squeeze(T, 1)

    X = l2_norm(X)
    
    return X, T

def detach_and_save(X, T, method):
    # Detach
    X = X.detach().cpu()
    T = T.detach().cpu()

    # Save arrays
    print('Saving embedding arrays...')
    
    np.save('X_{}_100_classes.npy'.format(method), X)
    np.save('T_{}_100_classes.npy'.format(method), T)

if args.compute_embeddings:
    x_base, t_base = computeXT(dl_ev, baseline_model)
    x_metrix, t_metrix = computeXT(dl_ev, metrix_model)

    detach_and_save(x_base, t_base, 'baseline')
    detach_and_save(x_metrix, t_metrix, 'metrix')

# Load arrays
print('Loading arrays...')
x_base = np.load('X_baseline_100_classes.npy')
t_base = np.load('T_baseline_100_classes.npy')

x_metrix = np.load('X_metrix_100_classes.npy')
t_metrix = np.load('T_metrix_100_classes.npy')

x_base = torch.from_numpy(x_base)
t_base = torch.from_numpy(t_base)
x_metrix = torch.from_numpy(x_metrix)
t_metrix = torch.from_numpy(t_metrix)

# Choose the number of neighbors to retrieve
K = 5

# Calculate the similarity matrices
cos_sim_base = F.linear(x_base, x_base)
cos_sim_metrix = F.linear(x_metrix, x_metrix)

# this one here has the indices of the topk images 
# of the dataloader for each image in ascending order
topk_idx_base = cos_sim_base.topk(1 + K)[1][:,1:]
topk_idx_metrix = cos_sim_metrix.topk(1 + K)[1][:,1:]

# this one has the corresponding predicted labels
y_base = t_base[cos_sim_base.topk(1 + K)[1][:,1:]]
y_metrix = t_metrix[cos_sim_metrix.topk(1 + K)[1][:,1:]]

random.seed(args.seed)
randomlist = random.sample(range(0, 5904), 5)

from PIL import Image, ImageDraw

for query in randomlist:
    print('For query with index {} the indices of the top 5 retrieved images of baseline model are: '.format(query), topk_idx_base[query])
    print('The query has label: ', t_base[query])
    print('And retrieved images have labels: ', y_base[query])
    print('For query with index {} the indices of the top 5 retrieved images of metrix model are: '.format(query), topk_idx_metrix[query])
    print('And retrieved images have labels: ', y_metrix[query])
    
    base_results = []
    metrix_results = []

    query_image = Image.open(dl_ev.dataset.im_paths[query]).resize((224, 224), resample=Image.BILINEAR)
    query_image.save('query_{}_label_{}.jpg'.format(query, t_base[query].item()))
    
    # Save baseline's retrievals
    for i, retrieval in enumerate(topk_idx_base[query].tolist()):
        retrieved_image = Image.open(dl_ev.dataset.im_paths[retrieval]).resize((224, 224), resample=Image.BILINEAR)
        
        draw = ImageDraw.Draw(retrieved_image) 
        if t_base[query] == t_base[retrieval]:
            draw.rectangle((0, 0, 223, 223), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 223, 223), outline='red', width=8)
        retrieved_image.save('query_{}_label_{}_rank_{}_baseline_label_{}.jpg'.format(query, t_base[query].item(), i+1, y_base[query][i].item()))

    # Save metrix's retrievals
    for i, retrieval in enumerate(topk_idx_metrix[query].tolist()):
        retrieved_image = Image.open(dl_ev.dataset.im_paths[retrieval]).resize((224, 224), resample=Image.BILINEAR)
        
        draw = ImageDraw.Draw(retrieved_image) 
        if t_metrix[query] == t_metrix[retrieval]:
            draw.rectangle((0, 0, 223, 223), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 223, 223), outline='red', width=8)
        retrieved_image.save('query_{}_label_{}_rank_{}_metrix_label_{}.jpg'.format(query, t_metrix[query].item(), i+1, y_metrix[query][i].item()))
   
    
# Get the corresponding images from dataset
#dl_ev.dataset.__getitem__(0)

# Get the corresponding image paths
#dl_ev.dataset.im_paths[5903]

#y_few = y_few.float().cpu()

''' 
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

tsne = TSNE(n_components=args.n_components, perplexity=args.perplexity, n_iter=args.n_iter, init='pca', verbose=9)
tsne_proj = tsne.fit_transform(x_few)

#cmap = cm.get_cmap('tab20')

#if args.num_classes > 20:
cmap1 = cm.get_cmap('tab20')
cmap2 = cm.get_cmap('Set3')
cmap = np.vstack((cmap1, cmap2))

fig, ax = plt.subplots(figsize=(8,8))

for lab in range(args.num_classes):
    if lab < 20:
        i = 0
        color = lab
    else:
        i = 1
        color = lab - 20
    indices = t_few == lab+100
    ax.scatter(tsne_proj[indices, 0],
               tsne_proj[indices, 1],
               c=np.array(cmap[i][0](color)).reshape(1, 4),
               #label=CUBClassNames.IDS_TO_NAMES[lab],
               label=lab+100,
               alpha=0.5)

ax.legend(bbox_to_anchor=(1.15, 1.0), loc='upper right', prop={'size': 10}, markerscale=2)
plt.tight_layout()

# Save figure
plt.savefig('{}_{}_classes_{}_perplexity_{}.png'.format(args.method, args.num_classes, args.perplexity, args.mode))

'''
    
