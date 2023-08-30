@dataclass(frozen=True)
class ClassifierModelConfig:
    """Default (abstract) parameters for the classifier."""

    def get_classifier_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> Classifier:
        """Helper method to return the correct classifier from config."""
        raise NotImplementedError("Implement in subclass.")


@dataclass(frozen=True)
class CNN_3D_XYZ_ModelConfig(ClassifierModelConfig):
    """Parameters for the CNN_3D_XYZ_Classifier."""

    conv_channels: List[int]
    """List of channels for each convolutional layer. Length specifies number of layers."""

    mlp_hidden_layers: List[int]
    """List of hidden layer sizes for the MLP. Length specifies number of layers."""

    n_fingers: int = 4
    """Number of fingers."""

    def input_shape_from_fingertip_config(self, fingertip_config: UnionFingertipConfig):
        return [
            4,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        ]

    def get_classifier_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> Classifier:
        """Helper method to return the correct classifier from config."""

        input_shape = self.input_shape_from_fingertip_config(fingertip_config)
        return CNN_3D_XYZ_Classifier(
            input_shape=input_shape,
            n_fingers=fingertip_config.n_fingers,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


# TODO(pculbert): fix.
# UnionClassifierModelConfig = Union[
#     CNN_3D_XYZ_ModelConfig, ClassifierModelConfig
# ]  # Passing none here so union is valid.


@dataclass(frozen=True)
class CNN_2D_1D_ModelConfig(ClassifierModelConfig):
    """Parameters for the CNN_2D_1D_Classifier."""

    conv_2d_film_hidden_layers: List[int]
    mlp_hidden_layers: List[int]
    conditioning_dim: int = 7
    n_fingers: int = 4

    @classmethod
    def grid_shape_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> List[int]:
        return [
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        ]

    def get_classifier_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> Classifier:
        """Helper method to return the correct classifier from config."""

        return CNN_2D_1D_Classifier(
            grid_shape=self.grid_shape_from_fingertip_config(fingertip_config),
            n_fingers=self.n_fingers,
            conditioning_dim=self.conditioning_dim,
            conv_2d_film_hidden_layers=self.conv_2d_film_hidden_layers,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


UnionClassifierModelConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "cnn_3d_xyz": CNN_3D_XYZ_ModelConfig(
            conv_channels=[32, 64, 128], mlp_hidden_layers=[256, 256]
        ),
        "cnn_2d_1d": CNN_2D_1D_ModelConfig(
            conv_2d_film_hidden_layers=[256, 256], mlp_hidden_layers=[256, 256]
        ),
    }
)
