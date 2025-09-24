from typing import Optional

from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from configs.pe.pe_final import PELabelMask
from configs.base_cfg import Config
from explainer.cam_generator.grad_cam_2d_seq_generator import GradCam2DSeqGenerator

from pytorch_grad_cam import GradCAMPlusPlus


class ExplainerConfig(PELabelMask):
    def __init__(
        self,
        dev_cfg: Optional[Config] = None,
    ):
        if dev_cfg is not None:
            print("Load dev cfg")
            print(dev_cfg)
            self.refresh(dev_cfg)

        self.explainer_class = GradCam2DSeqGenerator

        self.grad_cam_targets = [BinaryClassifierOutputTarget(1)]
        self.cam = GradCAMPlusPlus
        self.on_grad_cam = True
        self.on_deploy = True

    def get_model_setups(self, fold=0):
        return {
            "model_name": self.model_name,
            "img_size": self.img_size,
            "in_chans": self.in_chans,
            "num_heads": self.num_heads,
            "feature_space_dims": self.feature_space_dims,
            "lstm_hidden_dims": self.lstm_hidden_dims,
            "output_dim": self.output_dim,
            "pretrained": self.pretrained,
            "on_deploy": True,
            "on_grad_cam": True,
        }
