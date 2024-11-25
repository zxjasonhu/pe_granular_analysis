import torch
import torch.nn as nn

from models.pe.base_model_stage2 import BaseModelStage2
from models.utils.initalization import _initialize_weights


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
        self.aux_classifier = nn.Sequential(
            nn.GELU(), nn.Dropout(0.3), nn.Linear(feature_space_dims, 1, bias=False)
        )

        _initialize_weights(self.aux_classifier)
        _initialize_weights(self.encoder_dense)
        _initialize_weights(self.classifier)

    def forward(self, x, lengths=None):
        # x: img shape (batch_size, slice_size, 3, 224, 224)
        batch_size, seq_len, _, _, _ = x.shape
        # reshape if necessary
        if len(x.shape) == 5:
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.encoder_stage1(x)
        x = self.encoder_dense(x)

        # aux classifier
        aux_x = self.aux_classifier(x)
        aux_x = aux_x.view(batch_size, seq_len)

        # reshape back
        x = x.view(batch_size, seq_len, -1)
        x = self.classifier(x, lengths=lengths)

        return x, aux_x


if __name__ == "__main__":
    from torchinfo import summary

    img_size = 224
    channels = 3
    model = ModelWithAuxClassifier(
        model_name="vit",
        img_size=img_size,
        in_chans=channels,
        num_heads=8,
        feature_space_dims=64,
        lstm_hidden_dims=64,
        output_dim=1,
        pretrained=False,
    )

    # print(model)
    input_tensor = torch.randn(2, 10, channels, img_size, img_size)
    output = model(input_tensor)
    print(output[0].shape, output[1].shape)
