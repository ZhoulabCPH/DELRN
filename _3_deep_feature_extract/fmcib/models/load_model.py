from collections import OrderedDict

import torch
from loguru import logger
from torch import nn


class LoadModel(nn.Module):
    """
    A class representing a loaded model.

    Args:
        trunk (nn.Module, optional): The trunk of the model. Defaults to None.
        weights_path (str, optional): The path to the weights file. Defaults to None.
        heads (list, optional): The list of head layers in the model. Defaults to [].

    Attributes:
        trunk (nn.Module): The trunk of the model.
        heads (nn.Sequential): The concatenated head layers of the model.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the model.
        load(weights): Load the pretrained model weights.
    """

    def __init__(self, trunk=None, weights_path=None, heads=[], actFunction='GELU',  device_num=0) -> None:
        """
        Initialize the model.

        Args:
            trunk (optional): The trunk of the model.
            weights_path (optional): The path to the weights file.
            heads (list, optional): A list of layer sizes for the heads of the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.trunk = trunk
        self.device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")
        head_layers = []
        for idx in range(len(heads) - 1):
            current_layers = []
            current_layers.append(nn.Linear(heads[idx], heads[idx + 1], bias=True))

            if idx != (len(heads) - 2):
                if actFunction == 'ReLU':
                    current_layers.append(nn.ReLU(inplace=True))
                elif actFunction == 'GELU':
                    current_layers.append(nn.GELU())

            head_layers.append(nn.Sequential(*current_layers))

        if len(head_layers):
            self.heads = nn.Sequential(*head_layers)
        else:
            self.heads = nn.Identity()

        if weights_path is not None:
            self.load(weights_path)

        self.sigmoid = nn.Sigmoid()
        self.softsign = nn.Softsign()
        self.Softplus = nn.Softplus()
        self.SiLU = nn.SiLU()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.permute(0, 4, 2, 3, 1).float()
        out = self.trunk(x)
        # heads_out = self.heads(out)
        heads_out = out
        for idx, layer in enumerate(self.heads):
            heads_out = layer(heads_out)
            # If this is the second last layer, capture its output
            if idx == len(self.heads) - 2:
                second_last_fc_out = heads_out
        out = self.sigmoid(heads_out)

        return out, second_last_fc_out

    def load(self, weights):

        pretrained_model = torch.load(weights, map_location=self.device)

        if "trunk_state_dict" in pretrained_model:  # Loading ViSSL pretrained model
            trained_trunk = pretrained_model["trunk_state_dict"]
            msg = self.trunk.load_state_dict(trained_trunk, strict=False)
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        if "state_dict" in pretrained_model:  # Loading Med3D pretrained model
            trained_model = pretrained_model["state_dict"]

            # match the keys (https://github.com/Project-MONAI/MONAI/issues/6811)
            weights = {key.replace("module.", ""): value for key, value in trained_model.items()}
            weights = {key.replace("model.trunk.", ""): value for key, value in trained_model.items()}
            msg = self.trunk.load_state_dict(weights, strict=False)
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

            weights = {key.replace("model.heads.", ""): value for key, value in trained_model.items()}
            msg = self.heads.load_state_dict(weights, strict=False)
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        # Load trained heads
        if "head_state_dict" in pretrained_model:
            trained_heads = pretrained_model["head_state_dict"]

            try:
                msg = self.heads.load_state_dict(trained_heads, strict=False)
            except Exception as e:
                logger.error(f"Failed to load trained heads with error {e}. This is expected if the models do not match!")
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        logger.info(f"Loaded pretrained model weights \n")
