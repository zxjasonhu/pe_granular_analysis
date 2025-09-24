from typing import Mapping, Any

import torch
import torch.nn as nn

from models.pe.base_model_stage2 import BaseModelStage2
from models.utils.initalization import _initialize_weights


def load_checkpoint(model, checkpoint, strict=False, verbose=True):
    import re  # Add import here to fix the unresolved reference

    if isinstance(checkpoint, str):
        if verbose:
            print(f"=> loading checkpoint '{checkpoint}'")
        checkpoint = torch.load(checkpoint, map_location="cpu")
    else:
        if verbose:
            print("=> loading checkpoint from state_dict")

    state_dict = checkpoint["model_state_dict"]

    for key in list(state_dict.keys()):
        if "encoder_stage1.encoder.head.fc.weight" in key:
            state_dict["encoder_dense.2.weight"] = state_dict.pop(key)

        elif "encoder_stage1.encoder.head.fc.bias" in key:
            state_dict["encoder_dense.2.bias"] = state_dict.pop(key)

        elif "encoder." in key:
            state_dict[key.replace("encoder.", "")] = state_dict.pop(key)

        elif "aux_classifier" in key:
            state_dict.pop(key)

    # Remove 'module.' prefix if present
    state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
    orig_state_dict = model.state_dict()

    # Step 2: Identify keys in checkpoint but not in model
    extra_keys = [k for k in state_dict.keys() if k not in orig_state_dict]
    if extra_keys and verbose:
        print(f"Found {len(extra_keys)} keys in checkpoint not in model:")
        if len(extra_keys) < 10:
            print(", ".join(extra_keys))
        else:
            print(", ".join(extra_keys[:5]) + "... and others")

    # Step 3: Delete extra keys from checkpoint
    for k in extra_keys:
        del state_dict[k]

    # Step 4: Check for size mismatches
    mismatched_keys = []
    for k, v in list(state_dict.items()):
        if k not in orig_state_dict:
            mismatched_keys.append(k)
            continue

        ori_size = orig_state_dict[k].size()
        if v.size() != ori_size:
            if verbose:
                print(
                    f"SKIPPING!!! Shape of {k} mismatched: checkpoint {v.size()} vs model {ori_size}"
                )
            mismatched_keys.append(k)

    # Delete mismatched keys
    for k in mismatched_keys:
        if k in state_dict:
            del state_dict[k]

    # Step 5: Load polished checkpoint
    model.load_state_dict(state_dict, strict=strict)

    # Check if all keys are matched after cleaning
    if verbose:
        missing_keys = [k for k in orig_state_dict.keys() if k not in state_dict]
        if missing_keys:
            print(
                f"Warning: {len(missing_keys)} keys in model were not found in checkpoint"
            )
            if len(missing_keys) < 10:
                print(", ".join(missing_keys))
            else:
                print(", ".join(missing_keys[:5]) + "... and others")
        else:
            print("=> All required model keys found in checkpoint after cleaning")

        print(f"=> Checkpoint loaded successfully ({len(state_dict)} parameters)")

    # Clean up memory
    del state_dict
    del orig_state_dict
    del checkpoint

    return mismatched_keys, extra_keys


class ModelWithAuxClassifier(BaseModelStage2):
    def __init__(
        self,
        model_name="coatnet",
        img_size=224,
        in_chans=3,
        num_heads=8,
        feature_space_dims=64,
        lstm_hidden_dims=64,
        output_dim=1,
        pretrained=False,
        on_deploy=False,
        on_grad_cam=False,
    ):
        super().__init__(
            model_name=model_name,
            img_size=img_size,
            in_chans=in_chans,
            num_heads=num_heads,
            feature_space_dims=feature_space_dims,
            lstm_hidden_dims=lstm_hidden_dims,
            output_dim=output_dim,
            pretrained=pretrained,
            on_grad_cam=on_grad_cam,
        )

        self.on_deploy = on_deploy

        _initialize_weights(self.encoder_dense)
        _initialize_weights(self.classifier)

        if not on_deploy:
            self.aux_classifier = nn.Sequential(
                nn.GELU(), nn.Dropout(0.3), nn.Linear(feature_space_dims, 1, bias=False)
            )
            _initialize_weights(self.aux_classifier)

    def forward(self, x, lengths=None):
        # x: img shape (batch_size, slice_size, 3, 224, 224)
        batch_size, seq_len, _, _, _ = x.shape
        # reshape if necessary
        if len(x.shape) == 5:
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.encoder_stage1(x)
        x = self.encoder_dense(x)

        if self.on_deploy:
            # reshape back
            x = x.view(batch_size, seq_len, -1)
            x = self.classifier(x, lengths=lengths)
            return x

        # aux classifier
        aux_x = self.aux_classifier(x)
        aux_x = aux_x.view(batch_size, seq_len)

        # reshape back
        x = x.view(batch_size, seq_len, -1)
        x = self.classifier(x, lengths=lengths)

        return x, aux_x

    def custom_load_from_checkpoint(self, checkpoint: Mapping[str, Any]):
        load_checkpoint(self, checkpoint)


if __name__ == "__main__":
    from torchinfo import summary

    img_size = 224
    channels = 3
    model = ModelWithAuxClassifier(
        model_name="coatnet",
        img_size=img_size,
        in_chans=channels,
        num_heads=8,
        feature_space_dims=64,
        lstm_hidden_dims=64,
        output_dim=1,
        length_mask=False,
        pretrained=False,
    )

    summary(
        model,
        input_size=(2, 32, channels, img_size, img_size),
        depth=4,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
    )