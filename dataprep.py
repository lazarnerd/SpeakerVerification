#!/usr/bin/python
# -*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

import argparse
import glob
import h5py
import hashlib
import os
import pdb
import soundfile
import subprocess
import tarfile
import time
import torch
import torchaudio
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
from threading import Thread


## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description="VoxCeleb downloader")

parser.add_argument("--save_path", type=str, default="data", help="Target directory")
parser.add_argument("--user", type=str, default="user", help="Username")
parser.add_argument("--password", type=str, default="pass", help="Password")

parser.add_argument(
    "--download", dest="download", action="store_true", help="Enable download"
)
parser.add_argument(
    "--extract", dest="extract", action="store_true", help="Enable extract"
)
parser.add_argument(
    "--convert", dest="convert", action="store_true", help="Enable convert"
)
parser.add_argument(
    "--augment",
    dest="augment",
    action="store_true",
    help="Download and extract augmentation files",
)
parser.add_argument(
    "--convert_to_h5", dest="convert_to_h5", action="store_true", help="Enable convert"
)

args = parser.parse_args()

## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


## ========== ===========
## Download with wget
## ========== ===========


def download_part(args, line):
    url = line.split()[0]
    md5gt = line.split()[1]
    outfile = url.split("/")[-1]

    ## Download files
    out = subprocess.call(
        "wget %s --user %s --password %s -O %s/%s"
        % (url, args.user, args.password, args.save_path, outfile),
        shell=True,
    )
    if out != 0:
        raise ValueError(
            "Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website."
            % url
        )

    ## Check MD5
    md5ck = md5("%s/%s" % (args.save_path, outfile))
    if md5ck == md5gt:
        print("Checksum successful %s." % outfile)
    else:
        raise Warning("Checksum failed %s." % outfile)


def download(args, lines):
    threads = []
    for line in lines:
        t = Thread(target=download_part, args=(args, line))
        t.start()
        threads.append(t)
        time.sleep(10)
    for t in threads:
        t.join()


## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args, lines):

    for line in lines:
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        ## Concatenate files
        out = subprocess.call(
            "cat %s/%s > %s/%s" % (args.save_path, infile, args.save_path, outfile),
            shell=True,
        )

        ## Check MD5
        md5ck = md5("%s/%s" % (args.save_path, outfile))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise Warning("Checksum failed %s." % outfile)

        out = subprocess.call("rm %s/%s" % (args.save_path, infile), shell=True)


## ========== ===========
## Extract zip files
## ========== ===========
def full_extract(args, fname):

    print("Extracting %s" % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, "r") as zf:
            zf.extractall(args.save_path)


## ========== ===========
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):

    print("Extracting %s" % fname)
    with ZipFile(fname, "r") as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)
            # pdb.set_trace()
            # zf.extractall(args.save_path)


## ========== ===========
## Convert
## ========== ===========
def convert(args):

    files = glob.glob("%s/voxceleb2/*/*/*.m4a" % args.save_path)
    files.sort()

    print("Converting files from AAC to WAV")
    for fname in tqdm(files):
        outfile = fname.replace(".m4a", ".wav")
        out = subprocess.call(
            "ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null"
            % (fname, outfile),
            shell=True,
        )
        if out != 0:
            raise ValueError("Conversion failed %s." % fname)

def convert_to_stft(y):
    transform = torchaudio.transforms.Spectrogram(
        #sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window_fn=torch.hamming_window,
    )
    return transform(y)

def get_data_and_labels(args):
    train_list = "./lists/voxceleb2.txt"
    train_path = os.path.join(args.save_path,"voxceleb2")
    data_list = []
    data_label = []

    for line in open(train_list):
        speaker_label, filename = line.strip().split()
        data_label.append(int(speaker_label.replace("id","")))
        data_list.append(os.path.join(train_path,filename))

    return data_list, data_label

def get_meta_data():
    meta_list = "./lists/meta_voxceleb2.txt"
    meta_data = {}
    feature_indexing = {"gender": {"m": 0, "f": 1}, "dataset": {"dev": 0, "test": 1}}
    for line in open(meta_list):
        speaker_label, _, gender, dataset = line.replace(",","").strip().split()
        speaker_label = int(speaker_label.replace("id",""))
        meta_data[speaker_label] = [feature_indexing["gender"][gender], feature_indexing["dataset"][dataset]]

    return meta_data

def convert_to_h5(args):  
    data_list, data_label = get_data_and_labels(args)
    meta_data = get_meta_data()

    # Generates Samples
    def sample_generator():
        for i, data in enumerate(data_list):
            waveform, _ = torchaudio.load(data)
            audio, _ = soundfile.read(data) # raw audio in h5 ebenfalls speichern
            speaker_id = data_label[i]
            yield i, waveform, audio, speaker_id

    n_samples = len(data_list)
    T = 0
    n_freq = 257

    # x: Sample Dataset containing the spectrograms
    # shape:  [T, n_freq]
    # T:      The sum of lengths from the time axes of all the spectrograms
    # n_freq: The number of frequency bins in the spectrograms 
    x = h5py.File(args.save_path+'x.hdf5', 'w').create_dataset("x", (T,n_freq), maxshape=(None, n_freq)) 
    # y: Label Dataset containing the speaker IDs, and indices of the samples
    # shape:  [n_samples, 3]
    # The first column is the speaker ID
    # The second column is the START index of the sample in the spectrogram (x) dataset
    # The third column is the END index of the sample in the spectrogram (x) dataset
    y = h5py.File(args.save_path+'/y.hdf5', 'w').create_dataset("y", (n_samples, 3))
    
    # Same for raw audio with exception that x_audio contains audio
    x_audio = h5py.File(args.save_path+'/x_audio.hdf5', 'w').create_dataset("x_audio", (0,), maxshape=(None,))
    y_audio = h5py.File(args.save_path+'/y_audio.hdf5', 'w').create_dataset("y", (n_samples, 3))
    
    meta = h5py.File(args.save_path+'/meta.hdf5', 'w').create_dataset("meta", (n_samples, 3))

    x_shape = (T,n_freq)
    x_audio_shape = (0,)

    samples = sample_generator()
    for i, waveform, audio, speaker_id in samples:
        spectrogram = convert_to_stft(waveform)[0].T # spectrogram shape: [T, n_freq]

        # Store spectrogram
        start_index = x_shape[0]
        end_index   = start_index + spectrogram.size()[0]
        x_shape = (end_index, n_freq)
        x.resize(x_shape)
        x[start_index:end_index,:] = spectrogram
        x_shape = (end_index+1, n_freq)
        y[i,:] = (speaker_id, start_index, end_index)
        
        # Store raw audio
        start_index = x_audio_shape[0]
        end_index = start_index + audio.shape[0]
        x_audio_shape = (end_index,)
        x_audio.resize(x_audio_shape)
        x_audio[start_index:end_index,] = audio
        x_audio_shape = (end_index+1,)
        y_audio[i,:] = (speaker_id, start_index, end_index)

        # Store meta
        speaker_id = int(y[i,0])
        meta[i,:] = (speaker_id, meta_data[speaker_id][0], meta_data[speaker_id][1])
        
        if i%500==0:
            print(f"Iteration {i}/{n_samples}")   

    # Store in h5_file for spectrogram and raw audio
    with h5py.File(args.save_path+"/h5_file_spectrogram_train.hdf5", "w") as f_dst:
        f_dst.create_dataset("x", data=x)
        f_dst.create_dataset("y", data=y)
        f_dst.create_dataset("meta", data=meta)
    with h5py.File(args.save_path+"/h5_file_audio_train.hdf5", "w") as f_dst:
        f_dst.create_dataset("x", data=x_audio)
        f_dst.create_dataset("y", data=y_audio)
        f_dst.create_dataset("meta", data=meta)
            
    print("h5_files saved.")


## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):

    files = glob.glob("%s/musan/*/*/*.wav" % args.save_path)

    audlen = 16000 * 5
    audstr = 16000 * 3

    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace("/musan/", "/musan_split/"))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + "/%05d.wav" % (st / fs), fs, aud[st : st + audlen])

        print(idx, file)


## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":

    if not os.path.exists(args.save_path):
        raise ValueError("Target directory does not exist.")

    if args.augment:
        f = open("lists/augment.txt", "r")
        augfiles = f.readlines()
        f.close()

        download(args, augfiles)
        print("part-extract for simulated rir is running ...")
        part_extract(
            args,
            os.path.join(args.save_path, "rirs_noises.zip"),
            [
                "RIRS_NOISES/simulated_rirs/mediumroom",
                "RIRS_NOISES/simulated_rirs/smallroom",
            ],
        )
        print("full-extract for musan is running ...")
        full_extract(args, os.path.join(args.save_path, "musan.tar.gz"))
        print("splitting musan")
        split_musan(args)
        print("Done.")

    if args.download:
        f = open("lists/fileparts.txt", "r")
        fileparts = f.readlines()
        f.close()
        
        download(args, fileparts)

    if args.extract:
        f = open("lists/files.txt", "r")
        files = f.readlines()
        f.close()

        concatenate(args, files)
        for file in files:
            full_extract(args, os.path.join(args.save_path, file.split()[1]))
        out = subprocess.call(
            "mv %s/dev/aac/* %s/aac/ && rm -r %s/dev"
            % (args.save_path, args.save_path, args.save_path),
            shell=True,
        )
        out = subprocess.call(
            "mv %s/wav %s/voxceleb1" % (args.save_path, args.save_path), shell=True
        )
        out = subprocess.call(
            "mv %s/aac %s/voxceleb2" % (args.save_path, args.save_path), shell=True
        )

    if args.convert:
        convert(args)

    if args.convert_to_h5:
        convert_to_h5(args)
