import time
import deeplake
import random
from pathlib import Path

ROOT = Path("/workspaces/SpeakerVerification")

DATA = ROOT / "data"

VOXCELEB1 = DATA / "voxceleb1"
VOXCELEB2 = DATA / "voxceleb2"

VOXCELEB1_DEEPLAKE = (
    VOXCELEB1 / "deeplake" / "MEL__F_512__M_40__W_0.025s__H_0.01s__D_4.00s"
)
VOXCELEB2_DEEPLAKE = (
    VOXCELEB2 / "deeplake" / "MEL__F_512__M_40__W_0.025s__H_0.01s__D_2.00s"
)

VOXCELEB1_H5 = VOXCELEB1 / "h5" / "MEL__F_512__M_40__W_0.025s__H_0.01s__D_4.00s.hdf5"
VOXCELEB2_H5 = VOXCELEB2 / "h5" / "MEL__F_512__M_40__W_0.025s__H_0.01s__D_2.00s.hdf5"

# vox1_dataset = deeplake.load(str(VOXCELEB2_DEEPLAKE))
# vox1_dataset.rechunk(num_workers=8, scheduler="processed", progressbar=True)


batch_size = 800
num_workers = 8
num_epochs = 10
num_prefetch = 20

sample_length = 200


dataset = deeplake.load(str(VOXCELEB2_DEEPLAKE))


def transform(spectrogram):
    start = random.randint(0, spectrogram.shape[1] - sample_length)
    return spectrogram[:, start : start + sample_length]

dataloader = dataset.dataloader()
dataloader = dataloader.shuffle()
dataloader = dataloader.

loader = dataset.pytorch(
    num_workers=num_workers,
    shuffle=True,
    transform={"spectrograms": transform, "labels": None},
    batch_size=batch_size,
    pin_memory=True,
    progressbar=True,
    prefetch_factor=num_prefetch,
)

epoch_times = []
shuffle_buffer = []
start = time.time()
for epoch in range(num_epochs):
    start_epoch = time.time()
    for i, data in enumerate(loader):
        if i == 0:
            shuffle_buffer.append(time.time() - start_epoch)
    epoch_times.append(time.time() - start_epoch)
    print()
    print(f"Shuffle Buffer: {shuffle_buffer[-1]:6.2f}s")
    print(f"Epoch:          {epoch_times[-1]:6.2f}s")
    print()
    print()
print("============================================")
print(f"Total time: {time.time() - start:6.2f}s")
print(f"Avg time:   {sum(epoch_times) / len(epoch_times):6.2f}s")
print(f"Min time:   {min(epoch_times):6.2f}s")
print(f"Max time:   {max(epoch_times):6.2f}s")
