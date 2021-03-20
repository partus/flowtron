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

text = prepate_text("This is some new sentence of some long long words which apear to be quite long but not too long enoug. And repeat: This is some new sentence of some long long words which apear to be quite long but not too long enoug. And repeat: This is some new sentence of some long long words which apear to be quite long but not too long enoug.")
speaker_vecs = torch.tensor(5)[None,None].cuda()
with torch.no_grad():
    residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
    mels, attentions = model.infer(
        residual, speaker_vecs, text, gate_threshold=gate_threshold)

    print(mels.shape)




play_mels(mels)
output_dir = "exp_results"



import sounddevice as sd
def play_mels(mels):
    with torch.no_grad():
        audio = denoiser(waveglow.infer(mels, sigma=sigma), 0.01)

    audio = audio[0].cpu().numpy()
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    sd.play(audio[0], 24000)




print(audio.shape)

write(os.path.join(output_dir, 'sid{}_sigma{}.wav'.format(speaker_id, sigma)),
      data_config['sampling_rate'], audio[0])
