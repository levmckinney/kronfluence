"""Implementation of sharded storage inspired by FSDP."""

from dataclasses import dataclass
from enum import Enum
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard, Partial
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.futures import Future

from kronfluence.utils.state import State, release_memory


class ReduceStratagy(Enum):
    """Strategy for reducing a sharded buffer."""
    NONE = "none"
    ACCUMULATE = "accumulate"


@dataclass
class BufferConfig:
    """Configuration for a sharded buffer."""
    shard: bool = False
    materialize_on_forward: bool = False
    materialize_on_backward: bool = False
    shard_dim: int = 0


class BufferState(Enum):
    """State of a sharded buffer."""
    UNINITIALIZED = "uninitialized"
    NEEDS_SHARDING = "initialized_needs_sharding"
    SHARDED = "sharded"
    REPLICATED = "replicated"
    LOCAL = "local"

class ShardedStorage:
    """Sharded storage for PyTorch tensors."""

    buffer_configs: dict[str, BufferConfig]
    sharded_buffers: dict[str, DTensor]
    unsharded_buffers: dict[str, Tensor | DTensor]
    buffer_states: dict[str, BufferState]

    def __init__(self, state: State):
        self.buffer_configs = {}
        self.sharded_buffers = {}
        self.unsharded_buffers = {}
        self.buffer_states = {}
        self.state = state if state is not None else State()
        self.mesh = DeviceMesh() # TODO initialize mesh

    def register_buffer(self, name: str, config: BufferConfig):
        self.buffer_configs[name] = config
        self.buffer_states[name] = BufferState.UNINITIALIZED

    def is_initialized(self, key: str) -> bool:
        return self.buffer_states[key] != BufferState.UNINITIALIZED

    def __contains__(self, key: str) -> bool:
        # Assert key in config iff key in states
        assert key in self.buffer_configs == key in self.buffer_states, "Invariant violated!"
        return key in self.buffer_configs

    def __getitem__(self, key: str) -> Tensor | DTensor | None:
        if key not in self:
            raise KeyError(f"Buffer {key} is not registered.")
        
        match self.buffer_states[key]:
            case BufferState.SHARDED:
                raise ValueError(
                    f"Buffer {key} is sharded with placements {self.unsharded_buffers[key].placements}."
                    "Please ensure this buffer is materialized on the local process."
                )
            case BufferState.LOCAL | BufferState.NEEDS_SHARDING:
                assert not isinstance(self.unsharded_buffers[key], DTensor), "Invariant violated!"
                return self.unsharded_buffers[key]
            case BufferState.UNINITIALIZED:
                return None
            case BufferState.REPLICATED:
                return self.unsharded_buffers[key]
            case _:
                raise ValueError(f"Invalid buffer state: {self.buffer_states[key]}")

    def __delitem__(self, key: str):
        if key not in self:
            raise KeyError(f"Buffer {key} is not registered.")

        if self.buffer_states[key] != BufferState.UNINITIALIZED:
            del self.unsharded_buffers[key]
            if key in self.sharded_buffers:
                del self.sharded_buffers[key]
            if self.buffer_states[key] in [BufferState.SHARDED, BufferState.REPLICATED]:
                self.state.wait_for_everyone()
            release_memory()
            self.buffer_states[key] = BufferState.UNINITIALIZED

    def __setitem__(self, key: str, value: Tensor | None):
        if key not in self:
            raise KeyError(f"Buffer {key} is not registered.")

        # Delete buffer
        assert not isinstance(value, DTensor), "Sharded storage only supports tensors."
        del self[key]

        # setting to None just deletes the buffer
        if value is None:
            return

        # Initialize buffer again
        self.unsharded_buffers[key] = value

        if self.buffer_configs[key].shard:
            self.buffer_states[key] = BufferState.NEEDS_SHARDING
        else:
            self.buffer_states[key] = BufferState.LOCAL


    def _materialize(self, key: str) -> None:
        if key not in self:
            raise KeyError(f"Buffer {key} is not registered.")

        match self.buffer_states[key]:
            case (
                    BufferState.REPLICATED 
                    | BufferState.LOCAL 
                    | BufferState.UNINITIALIZED 
                    | BufferState.NEEDS_SHARDING
                ):
                return
            case BufferState.SHARDED:
                sharded_tensor = self.sharded_buffers[key]
                replicated_tensor = sharded_tensor.redistribute(
                    self.mesh,
                    placements=[Replicate()],
                )
                self.buffer_states[key] = BufferState.REPLICATED
                self.unsharded_buffers[key] = replicated_tensor
            case _:
                raise ValueError(f"Invalid buffer state: {self.buffer_states[key]}")


    def _dematerialize(self, key: str) -> None:
        match self.buffer_states[key]:
            case BufferState.UNINITIALIZED | BufferState.LOCAL | BufferState.SHARDED:
                return
            case BufferState.REPLICATED:
                del self.unsharded_buffers[key]
                self.state.wait_for_everyone()
                release_memory()
                self.buffer_states[key] = BufferState.SHARDED
            case BufferState.NEEDS_SHARDING:
                config = self.buffer_configs[key]
                assert not isinstance(self.unsharded_buffers[key], DTensor), "Invariant violated!"
                sharded_tensor = DTensor.from_local(
                    self.unsharded_buffers[key],
                    placements=[Shard(config.shard_dim)],
                )
                self.buffer_states[key] = BufferState.SHARDED
                self.sharded_buffers[key] = sharded_tensor

    def materialize_all(self) -> None:
        for key in self.buffer_configs:
            self._materialize(key)

    def dematerialize_all(self) -> None:
        for key in self.buffer_configs:
            self._dematerialize(key)
