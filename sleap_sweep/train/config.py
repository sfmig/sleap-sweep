from pathlib import Path

from sleap.nn.config import (  # CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
    TrainingJobConfig,
    UNetConfig,
)

TRAINING_SLP_FILE = (
    Path(__file__).parents[3]
    / "data"
    / "drosophila-melanogaster-courtship"
    / "courtship_labels.slp"
)


def create_cfg(hparam):
    # Existing configs can also be loaded from a .json file with:
    # cfg = sleap.load_config("training_config.json")
    # - initial_learning_rate
    # - batch_size?
    # - epochs

    # Multi-animal bottom-up model:
    # baseline_medium_rf.bottomup.json
    # - sigma for nodes
    # - sigma for edges
    # - receptive field size

    # See https://github.com/talmolab/sleap/blob/c8e3cd09bc6325ff18e2ad352b705f6175880263/tests/nn/test_training.py#L212

    # Initialize the default training job configuration.
    cfg = TrainingJobConfig()  # what is the default?

    # Update path to training data we just downloaded.
    cfg.data.labels.training_labels = str(TRAINING_SLP_FILE)
    cfg.data.labels.validation_fraction = 0.1

    # Preprocessing and training parameters.
    # cfg.data.instance_cropping.center_on_part = "thorax"
    cfg.optimization.augmentation_config.rotate = True
    cfg.optimization.initial_learning_rate = hparam["initial_learning_rate"]
    cfg.optimization.epochs = (
        1  # This is the maximum number of training rounds.
    )

    # These configures the actual neural network and the model type:
    cfg.model.backbone.unet = UNetConfig(filters=16, output_stride=4)
    cfg.model.heads.multi_instance = MultiInstanceConfig(
        confmaps=MultiInstanceConfmapsHeadConfig(
            output_stride=1, offset_refinement=False
        ),
        pafs=PartAffinityFieldsHeadConfig(output_stride=2),
    )
    # cfg.model.heads.centroid_instance = None
    # cfg.model.heads.centered_instance = CenteredInstanceConfmapsHeadConfig(
    #     anchor_part="thorax",
    #     sigma=1.5,
    #     output_stride=4
    # )

    # Setup how we want to save the trained model.
    # cfg.outputs.run_name = "baseline_model.topdown"
    cfg.outputs.run_name = "baseline_model.bottomup"

    return cfg
