```python

# 1. store data
# T => total sample length of ALL samples 
shape=[T, num_freqs]
shape=[0, 257]
torchfb = torchaudio.transforms.Spectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window)
for i, (spkr, sample) in enumerate(samples):
    spec = torchfb(sample)
    start = shape[0]
    shape[0] += spec.shape[0]
    x.set_shape(shape)    
    x[start:shape[1],:] = spectrogram[:,:]

    y[i,:] = (spkr, start, shape[0])




# data loading
indices = random.sample(100, len(num_samples))
for i in indices:
    (speaker_id, start, end) = y[i, :]
    sample_len = 100
    duration = end - start
    start = random.randint(duration-sample_len) + sample_len
    sample = x[start:start+sample_len, :]
```