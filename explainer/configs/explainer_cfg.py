from typing import Optional

from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from configs.base_cfg import Config
from data.pe.pe_dataset_ct_volume import PECTVolumeDataset
from pytorch_grad_cam import GradCAMPlusPlus
from explainer.cam_generator.grad_cam_2d_seq_generator import GradCam2DSeqGenerator


class ExplainerConfig(Config):
    def __init__(
        self,
        trail_id: str,
        path_to_experiment: Optional[str] = None,
        explainer_class: Optional[GradCam2DSeqGenerator.__class__] = None,
    ):
        assert trail_id is not None, "trail_id is None"
        if path_to_experiment is not None:
            self.path_to_experiment = path_to_experiment
        else:
            self.path_to_experiment = f"./experiments/{trail_id}"

        self.load(f"{self.path_to_experiment}/{trail_id}_config.yaml")
        self.trail_id = trail_id

        if explainer_class is None:
            self.explainer_class = GradCam2DSeqGenerator
        else:
            self.explainer_class = explainer_class

        self.best_model = "auc"
        self.dataset_class = PECTVolumeDataset
        self.grad_cam_targets = [BinaryClassifierOutputTarget(1)]
        self.cam = GradCAMPlusPlus

    def get_model_setups(self, fold=0):
        return {
            "encoder": "r152ir",
            "num_classes": 8,
            "pool": "max",
            "checkpoint_path": None,
            "maxpool_on_output": False,
        }


if __name__ == "__main__":
    import os

    print(os.getcwd())
    print(os.listdir())
    cfg = ExplainerConfig(path_to_experiment="../../configs/", trail_id="test")
    print(cfg)
    pass
