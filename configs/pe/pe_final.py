from configs.base_cfg import Config
from dataset.pe.pe_dataset_aux import PECTVolumeAuxLoss

from models.pe.aux_classifier import ModelWithAuxClassifier
from training.inference_with_aux import InferencerWithAuxLoss
from training.train_with_aux_mask import TrainerWithAuxLossMask


class PELabelMask(Config):
    working_dir = ""
    model_path = ""
    log_dir = ""
    test_result_dir = ""

    # model settings
    img_depth = 184
    img_size = 256

    pretrained = True
    model_head_path = ""

    num_heads = 8
    feature_space_dims = 64
    lstm_hidden_dims = 64

    in_chans = 3
    load_mask = False
    masks = []

    output_dim = 1
    length_mask = False
    load_stage1_path = None

    pad_seq_when_shorter = False
    model_name = "coatnet" # coatnet, efficientnet, resnet, vit

    # training settings
    batch_size = 2
    num_workers = 8

    data_parallel = False
    ddp = True

    weight_sampler = True

    target_label = "pe_present_in_exam"

    aux_loss_dataset_path = "not_assigned"
    aux_loss_target = "pe_present_on_image"

    model_class = ModelWithAuxClassifier
    dataset_class = PECTVolumeAuxLoss

    trainer = TrainerWithAuxLossMask
    inferencer = InferencerWithAuxLoss

    unfreeze_head_epochs = -1
    positive_weight = 1.0

    learning_rate = 2e-4
    real_batch_size = 32

    num_epochs = 30
    t_max = 30
    init_training_epochs = 15
    early_stopping_patience = 7

    eta_min = 5e-7
    weight_decay = 1e-4
    warmup_epochs = -1
    require_flip_inference = False

    main_loss_weight = {
        "phase1": 0.05,
    }

    early_stopping_target = "loss"


    training_dataset_path = None
    validation_dataset_path = None
    test_dataset_path = None

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
            "on_deploy": self.on_deploy,
            "on_grad_cam": self.on_grad_cam,
        }
