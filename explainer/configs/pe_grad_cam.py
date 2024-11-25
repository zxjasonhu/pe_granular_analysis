from typing import Optional

from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from data.pe.pe_dataset_ct_volume import PECTVolumeDataset
from explainer.configs.explainer_cfg import ExplainerConfig
from configs.pe.pe_final import PELabelMask
from explainer.cam_generator.grad_cam_2d_seq_generator import GradCam2DSeqGenerator

from pytorch_grad_cam import GradCAMPlusPlus

from models.pe.base_model_stage2 import BaseModelStage2


class PEGradCamConfig(ExplainerConfig, PELabelMask):
    def __init__(
        self,
        trail_id: str,
        path_to_experiment: Optional[str] = None,
        explainer_class: Optional[GradCam2DSeqGenerator.__class__] = None,
    ):
        super().__init__(trail_id, path_to_experiment, explainer_class)

        self.explainer_class = GradCam2DSeqGenerator
        self.best_model = "auc"
        self.model_class = BaseModelStage2
        self.dataset_class = PECTVolumeDataset
        self.grad_cam_targets = [BinaryClassifierOutputTarget(1)]
        self.cam = GradCAMPlusPlus

    def get_model_setups(self, fold=0):
        return {
            "img_size": self.img_size,
            "in_chans": self.in_chans,
            "num_heads": self.num_heads,
            "feature_space_dims": self.feature_space_dims,
            "lstm_hidden_dims": self.lstm_hidden_dims,
            "output_dim": self.output_dim,
            "length_mask": self.length_mask,
            "pretrained": self.pretrained,
        }


if __name__ == "__main__":
    import os

    print(os.getcwd())
    print(os.listdir())
    cfg = ExplainerConfig(path_to_experiment="../../configs/pe/", trail_id="test")
    print(cfg)
    pass
