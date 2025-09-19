import copy
import logging

import yaml


class Config:
    trail_id = "temp"
    phase = "phase1"  # starts from phase 1, 2, ...

    # lucky number:
    seed = 42

    # image processing:
    img_size = 224
    img_depth = 1  # number of slices in the image
    image_channels = 3  # raw image channels, 3 for RGB, 1 for grayscale etc.
    in_chans = 3  # total input channels, including masks
    window_center = 100
    window_width = 700
    xy_plane_enlarge_factor = 0.0  # enlarge by *(1.0 + xy_plane_enlarge_factor)
    z_axis_enlarge_factor = 0.0  # enlarge by *(1.0 + z_axis_enlarge_factor)
    pad_seq_when_shorter = False
    channel_strip = 1  # strip on the z axis for adjacent channels

    # data augmentation:
    apply_mixup = False

    # model training:
    trainer = None
    inferencer = None
    init_training_epochs = (
        0  # number of epochs to train the model from scratch before the early stopping
    )

    early_stopping_patience = 1000
    early_stopping_target = "loss"

    metrics_to_monitor = [
        # loss is automatically monitored by the trainer
        "AUC",
    ]  # AUC is AUROC if binary classification

    criterion = "BCEWithLogitsLoss"
    optimizer = "AdamW"
    batch_size = 64
    real_batch_size = 64  # if using gradient accumulation, this is the real batch size
    num_workers = 8
    num_epochs = 100
    ddp = False  # distributed data parallel
    data_parallel = False
    load_mask = False
    masks = None  # available masks in the project dataset

    learning_rate = 5e-4
    weight_decay = 1e-4
    warmup_epochs = 5
    t_max = 32
    eta_min = 5e-6
    clip_grad = 1.0

    fold_range = None  # default to train all folds
    current_fold = 0  # default fold number to train
    random_seq = False
    unfreeze_head_epochs = None

    # inference target:
    target_label = None
    require_flip_inference = False

    # model checkpointing:
    backbone = None
    resume_training = False
    working_dir = None
    model_path = None
    log_dir = None
    test_result_dir = None

    # model validation:
    folds = 5 # number of folds for cross-validation
    positive_weight = 1.0

    # model:
    weight_sampler = False
    training_dataset_path = None
    validation_dataset_path = None
    test_dataset_path = None
    on_deploy = False

    # auxiliary classifier:
    # main_loss_weight will be used if aux_loss exists.
    # if it is a float, always use this weight,
    # if it is a dict, use the corresponding weight in the dict for corresponding phase
    main_loss_weight = 1.0
    aux_loss_dataset_path = None
    aux_loss_target = None
    aux_regularizer = None

    # model class
    model_class = None

    # dataset class
    dataset_class = None

    # grad cam:
    on_grad_cam = False
    cam = None
    grad_cam_target_layers = None  # this should be specified in the model
    grad_cam_targets = None

    def to_dict(self):
        d = {}
        for k in dir(self):
            if not k.startswith("__"):
                if callable(getattr(self, k)):
                    # if getattr(self, k) is a class:
                    if isinstance(getattr(self, k), type):
                        d[k] = getattr(self, k)
                else:
                    d[k] = getattr(self, k)
        return d

    def __str__(self):
        return str(self.to_dict())

    def save(self, path):
        # save as yaml:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    # FIXME: not ready yet
    def load(self, path):
        # load from yaml:
        with open(path, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg_dict.items():
            setattr(self, k, v)

    def refresh(self, cfg):
        for k, v in cfg.to_dict().items():
            setattr(self, k, v)

    def get_model_setups(self, fold=0):
        logging.warning("get_model_setups() not implemented for base class")
        return {}

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    cfg = Config()
    for k, v in cfg.to_dict().items():
        print(k, ":", v)
