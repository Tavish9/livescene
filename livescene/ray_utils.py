""" custom ray Sampler for the livescene """

from typing import Callable, List, Optional, Tuple

import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import PDFSampler, Sampler, UniformLinDispPiecewiseSampler


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples.

    Args:
        num_proposal_samples_per_ray: Number of samples to generate per ray for each proposal step.
        num_nerf_samples_per_ray: Number of samples to generate per ray for the NERF model.
        num_proposal_network_iterations: Number of proposal network iterations to run.
        single_jitter: Use a same random jitter for all samples along a ray.
        update_sched: A function that takes the iteration number of steps between updates.
        initial_sampler: Sampler to use for the first iteration. Uses UniformLinDispPiecewise if not set.
    """

    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
        initial_sampler: Optional[Sampler] = None,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        if initial_sampler is None:
            self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = initial_sampler
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        lang_fns: Optional[List[Callable]] = None,
        density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []
        probability_list = []  # * get the probability for language nn training
        clip_feat_list = []  
        # deform_valid_list = []
        query_value_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            # print(f"*** i_level {i_level}, is_prop {is_prop}, updated {updated}")

            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples: RaySamples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
                if not is_prop:
                    # * final PDF: get times and construct ray_samplers with times if w/o is_prop
                    probability, clip_feat, times, query_value = lang_fns[i_level](ray_samples)
                    ray_samples.times = times
                    probability_list.append(probability)
                    clip_feat_list.append(clip_feat)
                    query_value_list.append(query_value)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    # * construct ray_samplers with times
                    probability, clip_feat, times, query_value = lang_fns[i_level](ray_samples)
                    ray_samples.times = times
                    density = density_fns[i_level](ray_samples.frustums.get_positions(), times)
                else:
                    with torch.no_grad():
                        # * construct ray_samplers with times
                        probability, clip_feat, times, query_value = lang_fns[i_level](ray_samples)
                        ray_samples.times = times
                        density = density_fns[i_level](ray_samples.frustums.get_positions(), times)

                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
                probability_list.append(probability)
                clip_feat_list.append(clip_feat)
                query_value_list.append(query_value)

        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list, probability_list, clip_feat_list, query_value_list
