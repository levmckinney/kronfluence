import pytest
import torch
from unittest.mock import MagicMock, patch

from kronfluence.utils.sharded_storage import (
    ShardedStorage,
    BufferConfig,
    BufferState,
)
from kronfluence.utils.state import State
from torch.distributed.tensor.placement_types import Replicate, Shard


@pytest.fixture
def mock_state():
    """Mocks the State object."""
    state = MagicMock(spec=State)
    state.device = torch.device("cpu")
    state.num_processes = 1
    state.wait_for_everyone = MagicMock()
    return state


@pytest.fixture
def sharded_storage(mock_state):
    """Provides a ShardedStorage instance with a mocked DeviceMesh."""
    with patch("kronfluence.utils.sharded_storage.DeviceMesh") as mock_device_mesh:
        mesh_instance = MagicMock()
        mock_device_mesh.return_value = mesh_instance
        storage = ShardedStorage(state=mock_state, mesh=mesh_instance)
        return storage


def test_register_buffer(sharded_storage):
    """Tests the registration of a buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("test_buffer", config)
    assert "test_buffer" in sharded_storage
    assert sharded_storage.buffer_states["test_buffer"] == BufferState.UNINITIALIZED
    assert not sharded_storage.is_initialized("test_buffer")
    assert sharded_storage["test_buffer"] is None


def test_set_and_get_local_buffer(sharded_storage):
    """Tests setting, getting, and deleting a non-sharded buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("local_buffer", config)
    tensor = torch.randn(10)
    sharded_storage["local_buffer"] = tensor

    assert sharded_storage.buffer_states["local_buffer"] == BufferState.LOCAL
    assert sharded_storage.is_initialized("local_buffer")
    assert torch.equal(sharded_storage["local_buffer"], tensor)

    # Test deletion by setting to None.
    sharded_storage["local_buffer"] = None
    assert sharded_storage.buffer_states["local_buffer"] == BufferState.UNINITIALIZED
    assert not sharded_storage.is_initialized("local_buffer")
    assert sharded_storage["local_buffer"] is None


def test_set_sharded_buffer(sharded_storage):
    """Tests setting a sharded buffer, which should put it in the NEEDS_SHARDING state."""
    config = BufferConfig(shard=True)
    sharded_storage.register_buffer("sharded_buffer", config)
    tensor = torch.randn(10)
    sharded_storage["sharded_buffer"] = tensor

    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.NEEDS_SHARDING
    assert torch.equal(sharded_storage["sharded_buffer"], tensor)


def test_delete_buffer(sharded_storage):
    """Tests the deletion of a buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("buffer_to_delete", config)
    tensor = torch.randn(10)
    sharded_storage["buffer_to_delete"] = tensor

    assert sharded_storage.is_initialized("buffer_to_delete")
    del sharded_storage["buffer_to_delete"]
    assert not sharded_storage.is_initialized("buffer_to_delete")
    assert sharded_storage.buffer_states["buffer_to_delete"] == BufferState.UNINITIALIZED


@patch("kronfluence.utils.sharded_storage.release_memory")
@patch("torch.distributed.tensor.DTensor.from_local")
def test_sharded_buffer_lifecycle(mock_from_local, mock_release_memory, sharded_storage, mock_state):
    """Tests the full state transition lifecycle of a sharded buffer."""
    mock_replicated_tensor = MagicMock()
    mock_from_local.return_value = mock_replicated_tensor

    mock_sharded_tensor = MagicMock()
    def redistribute_effect_replicated(mesh, placements):
        return mock_sharded_tensor

    def redistribute_effect_sharded(mesh, placements):
        return mock_replicated_tensor

    mock_replicated_tensor.redistribute.side_effect = redistribute_effect_replicated
    mock_sharded_tensor.redistribute.side_effect = redistribute_effect_sharded

    config = BufferConfig(shard=True, shard_dim=0)
    sharded_storage.register_buffer("sharded_buffer", config)
    tensor = torch.randn(20)

    # 1. Set tensor -> NEEDS_SHARDING
    sharded_storage["sharded_buffer"] = tensor
    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.NEEDS_SHARDING
    assert torch.equal(sharded_storage["sharded_buffer"], tensor)

    # 2. Dematerialize -> SHARDED
    sharded_storage.dematerialize_buffer("sharded_buffer")
    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.SHARDED
    mock_from_local.assert_called_once_with(
        tensor, device_mesh=sharded_storage.mesh, placements=[Replicate()]
    )
    mock_replicated_tensor.redistribute.assert_called_once_with(
        sharded_storage.mesh, placements=[Shard(config.shard_dim)]
    )
    assert sharded_storage.sharded_buffers["sharded_buffer"] is mock_sharded_tensor
    mock_state.wait_for_everyone.assert_called_once()
    mock_release_memory.assert_called_once()
    with pytest.raises(ValueError):
        _ = sharded_storage["sharded_buffer"]

    # 3. Materialize -> REPLICATED
    sharded_storage.materialize_buffer("sharded_buffer")
    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.REPLICATED
    mock_sharded_tensor.redistribute.assert_called_once_with(
        sharded_storage.mesh, placements=[Replicate()]
    )
    assert "sharded_buffer" in sharded_storage.unsharded_buffers

    # 4. Dematerialize again -> SHARDED
    mock_state.wait_for_everyone.reset_mock()
    mock_release_memory.reset_mock()
    sharded_storage.dematerialize_buffer("sharded_buffer")
    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.SHARDED
    assert "sharded_buffer" not in sharded_storage.unsharded_buffers
    mock_state.wait_for_everyone.assert_called_once()
    mock_release_memory.assert_called_once()

    # 5. Delete sharded buffer
    del sharded_storage["sharded_buffer"]
    assert not sharded_storage.is_initialized("sharded_buffer")


def test_materialize_all(sharded_storage):
    """Tests that materialize_all calls _materialize for each buffer."""
    sharded_storage.materialize_buffer = MagicMock()
    sharded_storage.register_buffer("buf1", BufferConfig())
    sharded_storage.register_buffer("buf2", BufferConfig())
    sharded_storage.materialize_all()
    assert sharded_storage.materialize_buffer.call_count == 2
    sharded_storage.materialize_buffer.assert_any_call("buf1")
    sharded_storage.materialize_buffer.assert_any_call("buf2")


def test_dematerialize_all(sharded_storage):
    """Tests that dematerialize_all calls _dematerialize for each buffer."""
    sharded_storage.dematerialize_buffer = MagicMock()
    sharded_storage.register_buffer("buf1", BufferConfig())
    sharded_storage.register_buffer("buf2", BufferConfig())
    sharded_storage.dematerialize_all()
    assert sharded_storage.dematerialize_buffer.call_count == 2
    sharded_storage.dematerialize_buffer.assert_any_call("buf1")
    sharded_storage.dematerialize_buffer.assert_any_call("buf2") 