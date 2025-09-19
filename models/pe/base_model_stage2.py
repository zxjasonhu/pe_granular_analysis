import timm
import torch
import torch.nn as nn

from models.utils.initalization import _initialize_weights


class NaiveSliceAttentionClassifier(nn.Module):
    def __init__(self, input_dim=64, num_heads=1, lstm_hidden_dims=64, output_dim=1):
        super(NaiveSliceAttentionClassifier, self).__init__()

        # Self-Attention Unit
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        # Inference details:
        # https://github.com/pytorch/pytorch/blob/74e6c877e9bc9c99466b1953f8a73b292d8e8ae7/aten/src/ATen/native/vulkan/ops/Lstm.cpp#L175
        self.lstm_layers = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dims,
            bidirectional=True,
            batch_first=True,
            num_layers=3,
            dropout=0.1,
        )

        # Final Dense Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dims, output_dim),  # 2x for bidirectional
        )

        # Gaussian Error Linear Unit (GELU) activation
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        # gelu activation and batch norm of input features
        x = self.gelu(x)
        # view to apply batch norm across input_dim
        x = self.batch_norm(x.view(-1, input_dim)).view(batch_size, seq_len, input_dim)

        # Self-Attention
        out, _ = self.self_attention(x, x, x)

        # dropout
        out = self.dropout(out)

        # residual connection
        out = self.layer_norm(out) + x

        # Pass through LSTM layers
        out, _ = self.lstm_layers(out)

        # Pooling: Using the output of the final timestep
        final_output = out[:, -1, :]  # should this be out[:, 0, :] instead?

        # dropout
        final_output = self.dropout(final_output)

        # Classification
        return self.classifier(final_output).squeeze(-1)


class BaseModelStage2(nn.Module):
    def __init__(
        self,
        model_name="coatnet",
        img_size=224,
        in_chans=3,
        num_heads=8,
        feature_space_dims=64,  # 768 for vit
        lstm_hidden_dims=64,
        output_dim=1,
        pretrained=False,
        on_grad_cam=False,
    ):
        super(BaseModelStage2, self).__init__()
        if "coatnet" in model_name:
            self.encoder_stage1 = timm.create_model(
                model_name="coatnet_0_rw_224.sw_in1k",
                # model_name="coatnet_2_rw_224.sw_in12k_ft_in1k",
                # model_name="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                img_size=img_size,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
            )
            encoded_num_features = self.encoder_stage1.num_features
        elif "resnet" in model_name:
            self.encoder_stage1 = timm.create_model(
                model_name="resnet50.a1_in1k",
                in_chans=in_chans,
                num_classes=0,
                features_only=False,
                drop_rate=0.2,
                drop_path_rate=0.2,
                pretrained=True,
            )
            encoded_num_features = self.encoder_stage1.layer4[-1].bn3.num_features
        elif "efficientnet" in model_name:
            self.encoder_stage1 = timm.create_model(
                model_name="tf_efficientnetv2_s.in21k",
                in_chans=in_chans,
                num_classes=0,
                features_only=False,
                drop_rate=0.2,
                drop_path_rate=0.2,
                pretrained=True,
            )
            encoded_num_features = self.encoder_stage1.conv_head.out_channels
        elif "vit" in model_name:
            self.encoder_stage1 = timm.create_model(
                model_name="vit_base_patch16_224",
                pretrained=True,
                num_classes=0,
                in_chans=in_chans,
            )
            encoded_num_features = 768

        self.encoder_dense = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(encoded_num_features, feature_space_dims),
        )

        self.classifier = NaiveSliceAttentionClassifier(
            input_dim=feature_space_dims,
            num_heads=num_heads,
            lstm_hidden_dims=lstm_hidden_dims,
            output_dim=output_dim,
        )

        # Initialize weights
        _initialize_weights(self.encoder_dense)
        _initialize_weights(self.classifier)

        # grad cam:
        if on_grad_cam:
            self.grad_cam_target_layers = [
                self.encoder_stage1.stages[-2].blocks[0].attn,
                # self.encoder_stage1.stages[0].blocks[0]
                self.encoder_stage1.stages[-1].blocks[-1].attn,
                # self.encoder_stage1.stages[-1].blocks[-2].attn#.proj,
            ]

    def forward(self, x, lengths=None):
        # reshape if necessary
        if len(x.shape) == 5:
            batch_size, depth, c, h, w = x.shape
            x = x.view(-1, c, h, w)
        elif len(x.shape) == 4:
            depth, c, h, w = x.shape
            batch_size = 1
        else:
            raise ValueError(
                f"Input shape must be either (B, D, C, H, W) or (D, C, H, W); got {x.shape}"
            )

        x = self.encoder_stage1(x)
        x = self.encoder_dense(x)

        # reshape back
        x = x.view(batch_size, depth, -1)
        x = self.classifier(x, lengths=lengths)

        if self.on_grad_cam:
            return x  # .squeeze(0)

        return x

    def load_head(self, path):
        checkpoint = torch.load(path)
        model_dict = checkpoint["model_state_dict"]
        for key in list(model_dict.keys()):
            if "encoder." in key:
                # remove "encoder." from the key
                model_dict[key[8:]] = model_dict.pop(key)

        for key in [
            "feature_ext.2.weight",
            "feature_ext.2.bias",
            "classifier.2.weight",
            "classifier.2.bias",
            "head.fc.weight",
            "head.fc.bias",
            "classifier.weight",
            "classifier.bias",
        ]:
            if key in model_dict:
                del model_dict[key]
        self.encoder_stage1.load_state_dict(model_dict)
        print(f"Loaded stage1 model from {path}")

    def load_prev_version(self, model_dict):
        print("Loading previous version of the model")
        for key in list(model_dict.keys()):
            if "encoder_stage1.encoder.head.fc.weight" in key:
                model_dict["encoder_dense.2.weight"] = model_dict.pop(key)

            elif "encoder_stage1.encoder.head.fc.bias" in key:
                model_dict["encoder_dense.2.bias"] = model_dict.pop(key)

            elif "encoder." in key:
                model_dict[key.replace("encoder.", "")] = model_dict.pop(key)

            elif "aux_classifier" in key:
                if self.on_grad_cam:
                    model_dict.pop(key)

        self.load_state_dict(model_dict)

    def freeze_head(self):
        for param in self.encoder_stage1.parameters():
            param.requires_grad = False
        self.encoder_stage1.train()

    def unfreeze_head(self):
        for param in self.encoder_stage1.parameters():
            param.requires_grad = True
        self.encoder_stage1.train()


if __name__ == "__main__":
    from torchinfo import summary

    img_size = 256
    length = 10
    in_chans = 4
    feature_size = 64
    input_shape = (1, length, in_chans, img_size, img_size)
    model = BaseModelStage2(
        img_size=img_size,
        in_chans=in_chans,
        num_heads=8,
        feature_space_dims=feature_size,
        lstm_hidden_dims=64,
        output_dim=1,
    )
    _input = torch.randn(input_shape)
    print(model(_input).shape)
    # print(model.encoder_stage1.stages[-1].blocks[-1])
    # from torchinfo import summary
    # summary(model, input_size=input_shape, device="cpu")
    pass
