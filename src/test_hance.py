# %%
from pathlib import Path
import hance
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

instruments = ["piano", "vocals", "bass", "drums"]
models = {instr:f"{instr}_separation.hance" for instr in instruments}


# %%
fn= Path("../../data/cruel_summer/cruel_summer.mp3")

x, sr= sf.read(fn)


# %%
x=x.astype(np.float32)
engine = hance.HanceEngine()

FORMAT = x.dtype 
CHANNELS = 1 if x.ndim==1 else x.shape[1] 
RATE = sr

# if CHANNELS>1:
#     x=np.mean(x, axis=1)
# %%

processors = {instr: engine.create_processor(models[instr], CHANNELS, RATE) for instr in instruments}

CHUNK = int(0.2*RATE)
Nsamples=x.shape[0]
k=0
ys={instr:[] for instr in instruments}
while k+CHUNK<Nsamples:
    print(f"\r {k}/{Nsamples}       ", end="")
    x_chunk=x[k:k+CHUNK]
    for instr in instruments:
        ys_chunk=processors[instr].process(x_chunk)
        assert ys_chunk.size>0, "Empty buffer!"
        ys[instr].append(ys_chunk)

    
    k+=CHUNK

# %%
for instr in instruments:
    ys[instr]=np.concatenate(ys[instr], axis=0)
# %%
idx_left=int(2*sr)
idx_right=int(2.5*sr)

def mono(x):
    if x.ndim==1:
        return x
    return np.max(x, axis=1)
tx=np.arange(x.shape[0])/RATE
plt.plot(tx[idx_left:idx_right],mono(x[idx_left:idx_right]))
y_total=mono(np.concatenate([ys[instr] for instr in instruments], axis=1))
ty=np.arange(y_total.shape[0])/RATE
plt.plot(ty[idx_left:idx_right], -np.abs(y_total[idx_left:idx_right]))    


#     ty=np.arange(ys[instr].shape[0])/RATE
#     plt.plot(ty, mono(ys[instr]), label=instr)

plt.legend()
# %%
for instr in instruments:
    sf.write(f"../../data/cruel_summer/cruel_summer_{instr}.wav", ys[instr], sr)
# %%
