from pathlib import Path
from monai.networks.nets import resnet50
import torch
from .load_model import LoadModel


def fmcib_model(eval_mode=True, heads=[], eval_path=None, actFunction='GELU',  device_num=0):
    trunk = resnet50(
        pretrained=False,
        n_input_channels=1,
        widen_factor=2,
        conv1_t_stride=2,
        feed_forward=False,
        bias_downsample=True,
    )
    current_path = Path(r'.\foundation_model_weights.torch')
    if eval_path is not None:
        eval_path = Path(eval_path)

    if not eval_mode:
        model = LoadModel(trunk=trunk, weights_path=current_path, heads=heads, device_num=device_num)
    else:
        model = LoadModel(trunk=trunk, weights_path=None, heads=heads, device_num=device_num)
    if eval_mode:
        print('load eval model')
        model.load_state_dict(torch.load(eval_path))
        model.eval()

    return model
