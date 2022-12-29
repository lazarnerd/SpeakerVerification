```python

#===================================================================================================
# 1. store data
#===================================================================================================

# sample generator to load the raw audio samples one by one
# important that this is a generator of some sort, so that you don't load
# all the samples into memory at once
def sample_generator():
    for i in range(n_samples):
        # code to load the raw audio sample & metadata
        yield i, raw_audio_sample, speaker_id

# Sample Dataset containing the spectrograms
# shape:  [T, n_freq]
# T:      The sum of lengths from the time axes of all the spectrograms
# n_freq: The number of frequency bins in the spectrograms 
x = h5_file['x']

# Label Dataset containing the speaker IDs, and indices of the samples
# shape:  [n_samples, 3]
# n_samples: The number of samples in the dataset
#
# The first column is the speaker ID
# The second column is the START index of the sample in the spectrogram (x) dataset
# The third column is the END index of the sample in the spectrogram (x) dataset
y = h5_file['y']
y.resize((n_samples, 3))

# Might also make sense to have meta datasets containing details about the speakers and the audio samples
# don't know what makes sense here to add, but it is useful to have this information in the dataset
meta_speakers = h5_file['meta_speakers']
meta_samples  = h5_file['meta_samples']

# The current shape of the spectrogram dataset (x)
# We don't know the length of each transformed sample beforehand,
# so we have to keep track of the current shape of the dataset and resize for each sample
x_dataset_shape = (0, n_freq)


# the torch function to convert the raw audio samples into spectrograms
# don't know what the output shape is, so I'm working with the assumption that it is [T, n_freq]
# where T is the length of the sample in time, and n_freq is the number of frequency bins
torchfb = torchaudio.transforms.Spectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    window_fn=torch.hamming_window
)

samples = sample_generator()
for i, raw_audio_sample, speaker_id in samples:
    spectrogram = torchfb(raw_audio_sample)
    # spectrogram shape: [T, n_freq]
    #
    # if the shape would be [n_freq, T], simply transpose it:
    # spectrogram = spectrogram.T

    start_index = x_dataset_shape[0]
    end_index   = start_index + spectrogram.shape[0]
    x_dataset_shape = (end_index, n_freq)

    # resize the spectrogram dataset (x) to fit the new sample
    x.resize(x_dataset_shape)

    # write the new sample to the dataset
    x[start_index:end_index, :] = spectrogram

    # write the label to the label dataset (y)
    y[i, :] = (speaker_id, start_index, end_index)

    # If you have additional metadata for each sample, write it to the meta_samples dataset
    # here to avoid iterating over the samples again later.
    

#===================================================================================================
# 2. data loading
#===================================================================================================

indices = random.sample(100, len(num_samples))
for i in indices:
    (speaker_id, start, end) = y[i, :]
    sample_len = 100
    duration = end - start
    start = random.randint(duration-sample_len) + sample_len
    sample = x[start:start+sample_len, :]
```