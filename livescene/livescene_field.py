"""
Fields for LiveScene
Adapted from kplanes_nerfstudio.kplanes.kplanes_field
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tinycudann as tcnn
import torch
from jaxtyping import Shaped
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import KPlanesEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from rich.console import Console
from torch import Tensor, nn

from livescene.livescene_encoding import LiveSceneEncoding

CONSOLE = Console(width=120)


def interpolate_ms_features(
    pts: torch.Tensor,
    grid_encodings: Iterable[KPlanesEncoding],
    concat_features: bool,
) -> torch.Tensor:
    """Combines/interpolates features across multiple dimensions and scales.

    Args:
        pts: Coordinates to query
        grid_encodings: Grid encodings to query
        concat_features: Whether to concatenate features at different scales

    Returns:
        Feature vectors
    """

    multi_scale_interp = [] if concat_features else 0.0
    for grid in grid_encodings:
        grid_features = grid(pts)

        if concat_features:
            multi_scale_interp.append(grid_features)
        else:
            multi_scale_interp = multi_scale_interp + grid_features

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp


class LiveSceneField(Field):
    """LiveScene field.

    Args:
        aabb: Parameters of scene aabb bounds
        num_images: How many images exist in the dataset
        geo_feat_dim: Dimension of 'geometry' features. Controls output dimension of sigma network
        grid_base_resolution: Base grid resolution
        grid_feature_dim: Dimension of feature vectors stored in grid
        concat_across_scales: Whether to concatenate features at different scales
        multiscale_res: Multiscale grid resolutions
        spatial_distortion: Spatial distortion to apply to the scene
        appearance_embedding_dim: Dimension of appearance embedding. Set to 0 to disable
        use_average_appearance_embedding: Whether to use average appearance embedding or zeros for inference
        linear_decoder: Whether to use a linear decoder instead of an MLP
        linear_decoder_layers: Number of layers in linear decoder
    """

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        geo_feat_dim: int = 15,  # TODO: This should be removed
        concat_across_scales: bool = True,  # TODO: Maybe this should be removed
        grid_base_resolution: Sequence[int] = (128, 128, 128),
        grid_feature_dim: int = 32,
        multiscale_res: Sequence[int] = (1, 2, 4),
        spatial_distortion: Optional[SpatialDistortion] = None,
        appearance_embedding_dim: int = 0,
        use_average_appearance_embedding: bool = True,
        linear_decoder: bool = False,
        linear_decoder_layers: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.num_images = num_images
        self.geo_feat_dim = geo_feat_dim
        self.grid_base_resolution = list(grid_base_resolution)
        self.concat_across_scales = concat_across_scales
        self.spatial_distortion = spatial_distortion
        self.linear_decoder = linear_decoder
        self.has_time_planes = len(grid_base_resolution) > 3

        # Init planes
        self.grids = nn.ModuleList()
        for res in multiscale_res:
            # Resolution fix: multi-res only on spatial planes
            resolution = [
                r * res for r in self.grid_base_resolution[:3]
            ] + self.grid_base_resolution[3:]
            self.grids.append(KPlanesEncoding(resolution, grid_feature_dim))
        self.feature_dim = (
            grid_feature_dim * len(multiscale_res)
            if self.concat_across_scales
            else grid_feature_dim
        )  # * True

        # Init appearance code-related parameters
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            assert (
                self.num_images is not None
            ), "'num_images' must not be None when using appearance embedding"
            self.appearance_ambedding = Embedding(
                self.num_images, self.appearance_embedding_dim
            )
            self.use_average_appearance_embedding = (
                use_average_appearance_embedding  # for test-time
            )

        # Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for combining the color
            # features into RGB.
            # Architecture based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
            in_dim_color = (
                self.direction_encoding.n_output_dims
                + self.geo_feat_dim
                + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = (
                SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0
            )

        if self.has_time_planes:
            assert (
                ray_samples.times is not None
            ), "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat(
                (positions, timestamps), dim=-1
            )  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat,
            grid_encodings=self.grids,
            concat_features=self.concat_across_scales,
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features).view(
                *ray_samples.frustums.shape, -1
            )
        else:
            features = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1
            )

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, features

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None

        output_shape = ray_samples.frustums.shape
        directions = ray_samples.frustums.directions.reshape(-1, 3)

        if self.linear_decoder:
            color_features = [density_embedding]
        else:
            directions = get_normalized_directions(directions)
            d = self.direction_encoding(directions)
            color_features = [d, density_embedding.view(-1, self.geo_feat_dim)]

        if self.appearance_embedding_dim > 0:
            if self.training:
                assert ray_samples.camera_indices is not None
                camera_indices = ray_samples.camera_indices.squeeze()
                embedded_appearance = self.appearance_ambedding(camera_indices)
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_ambedding.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                )

            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)
        if self.linear_decoder:
            basis_input = directions
            if self.appearance_ambedding_dim > 0:
                basis_input = torch.cat([directions, embedded_appearance], dim=-1)
            basis_values = self.color_basis(
                basis_input
            )  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(
                basis_input.shape[0], 3, -1
            )  # [batch, color_feature_len, 3]
            rgb = torch.sum(
                color_features[:, None, :] * basis_values, dim=-1
            )  # [batch, 3]
            rgb = torch.sigmoid(rgb).view(*output_shape, -1).to(directions)
        else:
            rgb = self.color_net(color_features).view(*output_shape, -1)

        return {FieldHeadNames.RGB: rgb}


class LiveSceneDensityField(Field):
    """A lightweight density field module.
    * with density only and w/o rgb

    Args:
        aabb: Parameters of scene aabb bounds
        resolution: Grid resolution
        num_output_coords: dimension of grid feature vectors
        spatial_distortion: Spatial distortion to apply to the scene
        linear_decoder: Whether to use a linear decoder instead of an MLP
    """

    def __init__(
        self,
        aabb: Tensor,
        density_feature_dim: int,
        density_resolution: List[int],
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.has_time_planes = len(density_resolution) > 3
        self.density_resolution = density_resolution
        self.density_feature_dim = density_feature_dim
        self.linear_decoder = linear_decoder
        self.grids = KPlanesEncoding(
            density_resolution, density_feature_dim, init_a=0.1, init_b=0.15
        )  # stay consistent with ProposalNetworkSampler.grids

        self.density_net = tcnn.Network(
            n_input_dims=density_feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if self.linear_decoder else "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        CONSOLE.log(
            f"Initialized LiveSceneDensityField. with time-planes={self.has_time_planes}"
        )

    # pylint: disable=arguments-differ
    def density_fn(
        self,
        positions: Shaped[Tensor, "*bs 3"],
        times: Optional[Shaped[Tensor, "*bs 1"]] = None,
    ) -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Overrides base function to add times in samples
        Args:
            positions: the origin of the samples/frustums
            times: the time of rays
        """
        if times is not None and (len(positions.shape) == 3 and len(times.shape) == 2):
            # position is [ray, sample, 3]; times is [ray, 1]
            times = times[:, None]  # RaySamples can handle the shape

        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = (
                SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0
            )

        if self.has_time_planes:
            assert (
                ray_samples.times is not None
            ), "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat(
                (positions, timestamps), dim=-1
            )  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat, grid_encodings=[self.grids], concat_features=False
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        density_before_activation = self.density_net(features).view(
            *ray_samples.frustums.shape, -1
        )
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, None


class LiveSceneLanguageField(Field):
    """A lightweight language field module.

    Args:
        aabb: Parameters of scene aabb bounds
        resolution: Grid resolution
        num_output_coords: dimension of grid feature vectors
        spatial_distortion: Spatial distortion to apply to the scene
        linear_decoder: Whether to use a linear decoder instead of an MLP
    """

    def __init__(
        self,
        aabb: Tensor,
        lang_feature_dim: int,
        lang_resolution: List[int],
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = False,
        use_gt_val: bool = False,
        ncls=1,
        max_prob=0.0,
        val_resolution=256,
        max_fids=1,
        clip_n_dims=512,
        val_feature_dim=4,
        val_neus=64,
        cat_embed=False,
        use_pe=True,
        clip_val=False,
        lang_neus=64,
        lang_nlys=1,
        train_with_lang=False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.lang_feature_dim = lang_feature_dim
        self.lang_resolution = lang_resolution
        self.linear_decoder = linear_decoder
        self.has_time_planes = len(lang_resolution) > 3
        self.grids = KPlanesEncoding(
            lang_resolution, lang_feature_dim, init_a=0.1, init_b=0.15
        )
        self.use_gt_val = use_gt_val
        self.ncls = ncls
        self.max_prob = max_prob
        self.use_pe = use_pe
        self.val_feature_dim = val_feature_dim
        self.cat_embed = cat_embed
        self.clip_val = clip_val
        self.max_fids = max_fids
        self.train_with_lang = train_with_lang

        if not self.use_gt_val:
            self.val_embeds = LiveSceneEncoding(
                resolution=[val_resolution], num_components=val_feature_dim
            )
            self.val_net = tcnn.Network(
                n_input_dims=val_feature_dim,
                n_output_dims=ncls,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "None" if self.linear_decoder else "ReLU",
                    "output_activation": "None" if clip_val else "Sigmoid",  # tanh
                    "n_neurons": val_neus,
                    "n_hidden_layers": 1,
                },
            )

        self.semantic_net = tcnn.Network(
            n_input_dims=lang_feature_dim + val_feature_dim
            if cat_embed
            else lang_feature_dim + ncls,
            n_output_dims=ncls + 1,  # for background
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if self.linear_decoder else "ReLU",
                "output_activation": "None",
                "n_neurons": lang_neus,
                "n_hidden_layers": lang_nlys,
            },
        )

        if self.train_with_lang:
            self.semantic_mapping = LiveSceneEncoding(
                resolution=[self.max_fids // 2] * ncls, num_components=clip_n_dims
            )

        CONSOLE.log(
            f"Initialized LiveSceneLanguageField. with time-planes={self.has_time_planes} lang_resolution={lang_resolution} - ncls={ncls}"
        )

    def pos_encoding(self, ids, scale=math.pi * 2):
        if self.use_pe:
            pe = torch.zeros(*ids.shape[:2], self.val_feature_dim, device=ids.device)
            div_term = torch.exp(
                torch.arange(0, self.val_feature_dim, 2).float()
                * (-math.log(1000.0) / self.val_feature_dim)
            ).to(ids.device)
            pe[..., 0::2] = torch.sin(ids * scale * div_term)
            pe[..., 1::2] = torch.cos(ids * scale * div_term)
            return pe.flatten(0, 1)
        else:
            return 0

    def seq_encoding(self, ids):
        return ids.flatten(0, 1)

    def lang_fn(
        self,
        ray_samples: RaySamples,
        **kwargs,
    ) -> Tensor:
        """Computes and returns the times (attribute value) coodanates."""
        positions = ray_samples.frustums.get_positions()
        positions_flat = positions.view(-1, positions.shape[-1])
        features = torch.sigmoid(
            interpolate_ms_features(
                positions_flat, grid_encodings=[self.grids], concat_features=False
            )
        )

        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)

        if not kwargs.get("viewer", False):
            # NOTE: sigmoid, clip, embeds
            if self.use_gt_val:
                query_values = ray_samples.metadata["atrb_val"].flatten(0, 1)[..., :-1]
            else:
                ids = ray_samples.metadata["id"]
                embeds = self.val_embeds(
                    ids.reshape(-1, ids.shape[-1])
                ).squeeze()  # [n_rays * n_samples, val_feature_dim]
                embeds = embeds + self.pos_encoding(ids, scale=math.pi * 2)
                if self.cat_embed:
                    query_values = self.val_net(
                        embeds.detach()
                    ).float()  # [n_rays * n_samples, ncls]
                else:
                    query_values = self.val_net(
                        embeds
                    ).float()  # [n_rays * n_samples, ncls]
            if self.training and self.train_with_lang:
                ids = ray_samples.metadata["id"]
                clip_embed = self.semantic_mapping(ids.reshape(-1, ids.shape[-1])).view(
                    *positions.shape[:-1], -1, self.semantic_mapping.get_out_dim()
                )  # [n_rays, n_samples, cls, clip_dim]
        else:
            values = ray_samples.metadata["atrb_val"]
            values = values.reshape(-1, values.shape[-1])
            query_values = values

        if self.clip_val:
            query_values = torch.clip(query_values, 0, 1)
        if self.cat_embed:
            probability = self.semantic_net(torch.cat([features, embeds], dim=-1)).view(
                *positions.shape[:-1], -1
            )  # [n_rays, n_samples, ncls+1]
        else:
            probability = self.semantic_net(
                torch.cat([features, query_values], dim=-1)
            ).view(*positions.shape[:-1], -1)  # [n_rays, n_samples, ncls+1]

        with torch.no_grad():
            softmax_probability = torch.softmax(probability, -1)
            max_probs, max_indices = torch.max(
                softmax_probability[..., :-1], dim=-1, keepdim=True
            )  # [n_rays, n_samples, 1]

        query_values = query_values.view(*positions.shape[:-1], -1)
        max_prb_values = torch.gather(
            query_values, -1, max_indices
        )  # [n_rays, n_samples, 1]
        valids = max_probs >= self.max_prob  # [n_rays, n_samples, 1]
        selected_values = torch.zeros_like(positions)[..., 0:1]
        selected_values[valids] = max_prb_values[valids]

        clip_feat = None
        if self.training and self.train_with_lang:
            clip_feat = torch.gather(
                clip_embed,
                -2,
                max_indices.unsqueeze(-1).expand(-1, -1, -1, clip_embed.shape[-1]),
            )  # [n_rays, n_samples, 1, clip_dim]
            clip_feat = (clip_feat / clip_feat.norm(dim=-1, keepdim=True)).squeeze()
        return probability, clip_feat, selected_values, query_values
