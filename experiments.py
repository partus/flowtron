import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import argparse
import json
import sys
import numpy as np
import torch


from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write
from denoiser import Denoiser
import matplotlib.pyplot as plt
%matplotlib inline
flowtron_path = "models/flowtron_ljs.pt"
flowtron_path = "models/flowtron_libritts2p3k.pt"
waveglow_path = "models/waveglow_256channels_universal_v5.pt"
text = "This is a standard text of known words in English, like dog, food, cat, love as well as unknown words like hypoblacmulish and birabadusavam."
speaker_id = 0
n_frames = 4000
sigma = 0.7
gate_threshold = 0.5
seed = 1234

with open("config.json") as f:
    data = f.read()

config = json.loads(data)
# update_params(config, args.params)

data_config = config["data_config"]
model_config = config["model_config"]
model_config["n_speakers"] = 2000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

waveglow = torch.load(waveglow_path)['model']
_ = waveglow.eval().cuda()
denoiser = Denoiser(waveglow).cuda().eval()

# load flowtron
model = Flowtron(**model_config).cuda()
from train import warmstart
model = warmstart(flowtron_path, model)

print("Loaded checkpoint '{}')" .format(flowtron_path))

def prepate_text(text):
    text = trainset.get_text(text).cuda()
    return text[None]
ignore_keys = ['training_files', 'validation_files']
trainset = Data(
    data_config['training_files'],
    **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))
speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()

speaker_vecs = speaker_vecs[None]




# text = prepate_text("This is some new sentence of some long long words which apear to be quite long but not too long enoug. And repeat: This is some new sentence of some long long words which apear to be quite long but not too long enoug. And repeat: This is some new sentence of some long long words which apear to be quite long but not too long enoug.")
text = prepate_text(text)
text = prepate_text("This is some short elaborated text")
text = prepate_text("In the following example, we show how to visualize large image datasets using UMAP. Here, we use load_digits, a subset of the famous MNIST dataset that was downsized to 8x8 and flattened to 64 dimensions. Although there iss over 1000 data points, and many more dimensions than the previous example, it is still extremely fast. This is because UMAP is optimized for speed, both from a theoretical perspective, and in the way it is implemented. Learn more in this comparison post.")
len(text[0])
def sp_tensor(id):
    return torch.tensor(id)[None,None].cuda()

def sp_vec(id):
    return model.speaker_embedding(sp_tensor(id))

import sounddevice as sd
def play_mels(mels):
    with torch.no_grad():
        audio = denoiser(waveglow.infer(mels, sigma=sigma), 0.01)

    audio = audio[0].cpu().numpy()
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    sd.play(audio[0], 24000)

speaker_vecs = sp_tensor(5)

sp_vec(5)


s_vec = (sp_vec(5)+sp_vec(0)+sp_vec(3))/3
s_vec = sp_vec(20)+sp_vec(5)
s_vec = sp_vec(26)
s_vec = sp_vec(59)
s_vec = (sp_vec(59)+sp_vec(0))/2
s_vec = sp_vec(120)
s_vec = -sp_vec(120)
plt.hist(s_vec.cpu().detach().numpy().ravel())
s_vec.shape
# s_vec1 = torch.rand([1,1,128]).cuda()*2
# s_vec1
# s_vec

residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
s_vec = torch.rand([1,1,128]).cuda()*2-1
s_vec = torch.ones([1,1,128]).cuda()-1
s_vec = sp_vec(5)
with torch.no_grad():
    mels, attentions = model.infer(
        residual, sp_tensor(0), text, gate_threshold=gate_threshold,speaker_vecs=s_vec)
play_mels(mels)
sd.stop()
output_dir = "exp_results"








print(audio.shape)

write(os.path.join(output_dir, 'sid{}_sigma{}.wav'.format(speaker_id, sigma)),
      data_config['sampling_rate'], audio[0])

emb = model.speaker_embedding.cpu()
emb = emb(torch.tensor(range(2000))).detach().numpy()

from sklearn.manifold import TSNE
from openTSNE import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_jobs = -1)
tsne_results = tsne.fit_transform(emb)
plt.scatter(*tsne_results.transpose())
plt.scatter(*tsne_results.transpose())
tsne_results
import pandas as pd
speakers = pd.read_csv("filelists/libritts_speakerinfo.txt",delimiter="|",comment=';',header=None)
speakers
speakers.to_numpy()[:,:2]

import plotly.express as px
from sklearn.datasets import load_digits
from umap import UMAP

digits = load_digits()

umap_2d = UMAP(random_state=0)
umap_2d.fit(digits.data)

projections = umap_2d.transform(digits.data)

fig = px.scatter(
    projections, x=0, y=1,
    color=digits.target.astype(str), labels={'color': 'digit'}
)
tsne_results
px.scatter(tsne_results,x=0, y=1)
fig.show()
