import re

import takpy as tak
import torch
from safetensors import safe_open

from models.baseline.model import Baseline
from models.baseline.representation import game_to_tensor, input_channels, policy_channels

tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():  # noqa: SIM118
        tensors[key] = f.get_tensor(key)

# for name, tensor in tensors.items():
#     print(name, tensor.shape)

# Renaming
state_dict = {}
for key, value in tensors.items():
    if "simhash" in key:
        continue
    if "ube" in key:
        continue

    s = key
    # Input
    s = s.replace("core.input_conv2d", "initial_conv.block.0")
    s = s.replace("core.batch_norm", "initial_conv.block.1")
    # Output
    s = s.replace("policy.conv2d", "policy_out")
    s = s.replace("value.conv2d", "value_conv.0")
    s = s.replace("value.linear", "value_fc.0")

    # Core
    s = re.sub(r"core\.res_block_(\d+)\.batch_norm\.(\w+)\_\_\d+", r"residual_blocks.\1.block.4.\2", s)
    s = re.sub(r"core\.res_block_(\d+)\.batch_norm\.(\w+)", r"residual_blocks.\1.block.1.\2", s)
    s = re.sub(r"core\.res_block_(\d+)\.conv2d\.(\w+)\_\_\d+", r"residual_blocks.\1.block.3.\2", s)
    s = re.sub(r"core\.res_block_(\d+)\.conv2d\.(\w+)", r"residual_blocks.\1.block.0.\2", s)

    state_dict[s] = value

N = 6
net = Baseline(input_channels(N), policy_channels(N), N)
net.load_state_dict(state_dict, strict=True)

# torch.save(net.state_dict(), "example.pt")
# x = torch.load("example.pt", weights_only=True)
# net.load_state_dict(x)


tps = "1,1,2,21,2,2/112S,2,2,21S,1,1/2,11121C,12,2S,2,1/1,211,1212,12,2,2/x,12,x,2112C,2,1/21,x3,1S,1121212S 2 47"
tensor = game_to_tensor(tak.game_from_tps(N, tps, 4))

batch_size = 1
with torch.no_grad():
    out = net.forward(tensor.repeat((batch_size, 1, 1, 1)))
print(out.policy.shape, out.value.shape)
# assert out.policy.shape == (batch_size, policy_channels(N), 6, 6)
# assert out.value.shape == (batch_size,)

print(f"### policy ###\n{out.policy}")
print(f"### value  ###\n{out.value}")


# import matplotlib.pyplot as plt
# import numpy as np
# import takpy as tak

# from models.baseline.representation import game_to_tensor

# tps = "2,x3,1,x/2,1,2,1S,1,2/2S,2121,11C,1,2,1/2,2S,2S,x,1112C,1/2,2,21,2,2,2/x2,21,x2,1 1 24"
# tensor = game_to_tensor(tak.game_from_tps(6, tps, 4))
# img = tensor.detach().numpy()
# img = np.permute_dims(img, axes=[1, 2, 0])
# layers = img.shape[2]

# assert layers == 6 * 6
# fig, axarr = plt.subplots(3, 15)
# for layer in range(3 * 15):
#     ax: plt.Axes = axarr[layer // 15][layer % 15]
#     if layer < layers:
#         x = img[:, :, layer]
#         x = np.flip(x, 0)
#         ax.imshow(x, vmin=0, vmax=1)
#         ax.axis("off")
#     else:
#         ax.remove()

# fig.subplots_adjust(left=None, bottom=None, right=None, top=None)
# plt.savefig("test.png")
