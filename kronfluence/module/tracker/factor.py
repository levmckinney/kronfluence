from typing import Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from kronfluence.factor.config import FactorConfig
from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    COVARIANCE_FACTOR_NAMES,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_ACTIVATION_COVARIANCE_PROCESSED,
    NUM_GRADIENT_COVARIANCE_PROCESSED,
    NUM_LAMBDA_PROCESSED,
)
from kronfluence.utils.exceptions import FactorsNotFoundError
from kronfluence.utils.sharded_storage import ShardedStorage


class CovarianceTracker(BaseTracker):
    """Tracks and computes activation and gradient covariance matrices for a given module."""

    def _update_activation_covariance_matrix(
        self, input_activation: torch.Tensor, count: Union[torch.Tensor, int]
    ) -> None:
        """Computes and updates the activation covariance matrix.

        Args:
            input_activation (torch.Tensor):
                The flattened input tensor to the module, provided by PyTorch's forward hook.
            count (int):
                The number of activations.
        """
        storage: ShardedStorage = self.module.storage
        if not storage.is_initialized(ACTIVATION_COVARIANCE_MATRIX_NAME):
            storage[NUM_ACTIVATION_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=count.device if isinstance(count, torch.Tensor) else None,
                requires_grad=False,
            )
            dimension = input_activation.size(1)
            storage[ACTIVATION_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=input_activation.dtype,
                device=input_activation.device,
                requires_grad=False,
            )
            storage.dematerialize_all()
        storage[NUM_ACTIVATION_COVARIANCE_PROCESSED] += count
        storage.accumulate(ACTIVATION_COVARIANCE_MATRIX_NAME, input_activation.t() @ input_activation)

    def _update_gradient_covariance_matrix(
        self, output_gradient: torch.Tensor, count: Union[torch.Tensor, int]
    ) -> None:
        """Computes and updates the pseudo-gradient covariance matrix.

        Args:
            output_gradient (torch.Tensor):
                The flattened gradient tensor with respect to the output of the module, provided
                by PyTorch's backward hook.
            count (int):
                The number of gradients.
        """
        storage: ShardedStorage = self.module.storage
        if not storage.is_initialized(GRADIENT_COVARIANCE_MATRIX_NAME):
            # In most cases, `NUM_GRADIENT_COVARIANCE_PROCESSED` and `NUM_ACTIVATION_COVARIANCE_PROCESSED` are
            # identical. However, they may differ when using gradient checkpointing or `torch.compile()`.
            storage[NUM_GRADIENT_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=count.device if isinstance(count, torch.Tensor) else None,
                requires_grad=False,
            )
            dimension = output_gradient.size(1)
            storage[GRADIENT_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=output_gradient.dtype,
                device=output_gradient.device,
                requires_grad=False,
            )
            storage.dematerialize_all()
        storage[NUM_GRADIENT_COVARIANCE_PROCESSED] += count
        alpha = 1
        if self.module.gradient_scale != 1.0:
            alpha = self.module.gradient_scale**2.0
        output_gradient = alpha * output_gradient
        storage.accumulate(GRADIENT_COVARIANCE_MATRIX_NAME, output_gradient.t() @ output_gradient)

    def register_hooks(self) -> None:
        """Sets up hooks to compute activation and gradient covariance matrices."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            input_activation = (
                inputs[0]
                .detach()
                .to(
                    dtype=self.module.factor_args.activation_covariance_dtype,
                    copy=self.module.attention_mask is not None,
                )
            )
            # Computes and updates activation covariance during forward pass.
            input_activation, count = self.module.get_flattened_activation(input_activation=input_activation)
            self._update_activation_covariance_matrix(input_activation=input_activation, count=count)
            self.cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.factor_args.gradient_covariance_dtype)
            # Computes and updates pseudo-gradient covariance during backward pass.
            output_gradient, count = self.module.get_flattened_gradient(output_gradient=output_gradient)
            self._update_gradient_covariance_matrix(output_gradient=output_gradient, count=count)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    def exist(self) -> bool:
        """Checks if both activation and gradient covariance matrices are available."""
        storage: ShardedStorage = self.module.storage
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            if not storage.is_initialized(covariance_factor_name):
                return False
        return True

    def synchronize(self, num_processes: int) -> None:
        """Aggregates covariance matrices across multiple devices or nodes in a distributed setting."""
        del num_processes
        storage: ShardedStorage = self.module.storage
        if dist.is_initialized() and torch.cuda.is_available() and self.exist():
            for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
                if storage.buffer_configs[covariance_factor_name].shard:
                    continue # Already syncronized

                storage[covariance_factor_name] = storage[covariance_factor_name].cuda()
                dist.reduce(
                    tensor=storage[covariance_factor_name],
                    op=dist.ReduceOp.SUM,
                    dst=0,
                )

    def release_memory(self) -> None:
        """Clears all covariance matrices from memory."""
        storage: ShardedStorage = self.module.storage
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            del storage[covariance_factor_name]


class LambdaTracker(BaseTracker):
    """Tracks and computes Lambda matrices for a given module."""

    def _eigendecomposition_results_exist(self) -> bool:
        """Checks if eigendecomposition results are available."""
        storage: ShardedStorage = self.module.storage
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            if not storage.is_initialized(eigen_factor_name):
                return False
        return True

    def _update_lambda_matrix(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes and updates the Lambda matrix using provided per-sample gradient.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample gradient tensor for the given batch.
        """
        batch_size = per_sample_gradient.size(0)

        storage: ShardedStorage = self.module.storage
        requires_eig = FactorConfig.CONFIGS[self.module.factor_args.strategy].requires_eigendecomposition_for_lambda

        if not storage.is_initialized(NUM_LAMBDA_PROCESSED):
            storage[NUM_LAMBDA_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                requires_grad=False,
            )
            storage[LAMBDA_MATRIX_NAME] = torch.zeros(
                size=(per_sample_gradient.size(1), per_sample_gradient.size(2)),
                dtype=per_sample_gradient.dtype,
                device=per_sample_gradient.device,
                requires_grad=False,
            )

            if requires_eig:
                if not self._eigendecomposition_results_exist():
                    raise FactorsNotFoundError(
                        f"The strategy {self.module.factor_args.strategy} requires eigendecomposition "
                        f"results for Lambda computations, but they are not found."
                    )
            storage.dematerialize_all()

        storage[NUM_LAMBDA_PROCESSED].add_(batch_size)

        if requires_eig:
            storage.materialize_buffer(ACTIVATION_EIGENVECTORS_NAME) # TODO: Overlap communication with computation better!
            storage.materialize_buffer(GRADIENT_EIGENVECTORS_NAME)
            if self.module.factor_args.use_iterative_lambda_aggregation:
                # This batch-wise iterative update can be useful when the GPU memory is limited.
                per_sample_gradient = torch.matmul(
                    per_sample_gradient,
                    storage[ACTIVATION_EIGENVECTORS_NAME],
                )
                for i in range(batch_size):
                    sqrt_lambda = torch.matmul(
                        storage[GRADIENT_EIGENVECTORS_NAME].t(),
                        per_sample_gradient[i],
                    )
                    storage.accumulate(LAMBDA_MATRIX_NAME, sqrt_lambda.square_())
            else:
                per_sample_gradient = (
                    torch.matmul(
                        storage[GRADIENT_EIGENVECTORS_NAME].t(),
                        torch.matmul(per_sample_gradient, storage[ACTIVATION_EIGENVECTORS_NAME]),
                    )
                    .square_()
                    .sum(dim=0)
                )
                storage.accumulate(LAMBDA_MATRIX_NAME, per_sample_gradient)
            storage.dematerialize_buffer(ACTIVATION_EIGENVECTORS_NAME)
            storage.dematerialize_buffer(GRADIENT_EIGENVECTORS_NAME)
        else:
            # Approximate the eigenbasis as identity.
            per_sample_gradient = per_sample_gradient.square_().sum(dim=0)
            storage.accumulate(LAMBDA_MATRIX_NAME, per_sample_gradient)

    def register_hooks(self) -> None:
        """Sets up hooks to compute lambda matrices."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach()
            device = "cpu" if self.module.factor_args.offload_activations_to_cpu else cached_activation.device
            cached_activation = cached_activation.to(
                device=device,
                dtype=self.module.factor_args.per_sample_gradient_dtype,
                copy=True,
            )
            if self.module.factor_args.has_shared_parameters:
                if self.cached_activations is None:
                    self.cached_activations = []
                self.cached_activations.append(cached_activation)
            else:
                self.cached_activations = cached_activation
            self.cached_hooks.append(
                outputs.register_hook(
                    shared_backward_hook if self.module.factor_args.has_shared_parameters else backward_hook
                )
            )

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.factor_args.per_sample_gradient_dtype)
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=self.cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient,
            ).to(dtype=self.module.factor_args.lambda_dtype)
            self.clear_all_cache()
            del output_gradient
            if self.module.gradient_scale != 1.0:
                per_sample_gradient.mul_(self.module.gradient_scale)
            # Computes and updates lambda matrix during backward pass.
            self._update_lambda_matrix(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.factor_args.per_sample_gradient_dtype)
            cached_activation = self.cached_activations.pop()
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient,
            )
            if self.cached_per_sample_gradient is None:
                self.cached_per_sample_gradient = torch.zeros_like(per_sample_gradient, requires_grad=False)
            # Aggregates per-sample gradients during backward pass.
            self.cached_per_sample_gradient.add_(per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_iteration(self) -> None:
        """Updates Lambda matrix using cached per-sample gradients."""
        if self.module.factor_args.has_shared_parameters:
            self.cached_per_sample_gradient = self.cached_per_sample_gradient.to(
                dtype=self.module.factor_args.lambda_dtype
            )
            if self.module.gradient_scale != 1.0:
                self.cached_per_sample_gradient.mul_(self.module.gradient_scale)
            self._update_lambda_matrix(per_sample_gradient=self.cached_per_sample_gradient)
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if Lambda matrices are available."""
        storage: ShardedStorage = self.module.storage
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            if not storage.is_initialized(lambda_factor_name):
                return False
        return True

    def synchronize(self, num_processes: int) -> None:
        """Aggregates Lambda matrices across multiple devices or nodes in a distributed setting."""
        del num_processes
        storage: ShardedStorage = self.module.storage
        if dist.is_initialized() and torch.cuda.is_available() and self.exist():
            for lambda_factor_name in LAMBDA_FACTOR_NAMES:
                if storage.buffer_configs[lambda_factor_name].shard:
                    continue # Already syncronized

                storage[lambda_factor_name] = storage[lambda_factor_name].cuda()
                dist.reduce(
                    tensor=storage[lambda_factor_name],
                    op=dist.ReduceOp.SUM,
                    dst=0,
                )

    def release_memory(self) -> None:
        """Clears Lambda matrices from memory."""
        self.clear_all_cache()
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            self.module.storage[lambda_factor_name] = None
