import pytest
import torch
from unittest.mock import MagicMock, patch

from kronfluence.utils.sharded_storage import (
    ShardedStorage,
    BufferConfig,
    BufferState,
)
from kronfluence.utils.state import State
from torch.distributed.tensor.placement_types import Replicate, Shard, Partial


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


def test_set_and_get_local_buffer_pytree(sharded_storage):
    """Tests setting, getting, and deleting a non-sharded PyTree buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("local_pytree", config)
    
    # Test with dict PyTree
    pytree_dict = {
        "tensor1": torch.randn(10),
        "tensor2": torch.randn(5, 5),
        "nested": {
            "tensor3": torch.randn(3, 3)
        }
    }
    sharded_storage["local_pytree"] = pytree_dict

    assert sharded_storage.buffer_states["local_pytree"] == BufferState.LOCAL
    assert sharded_storage.is_initialized("local_pytree")
    
    # Verify the structure is preserved
    retrieved = sharded_storage["local_pytree"]
    assert torch.equal(retrieved["tensor1"], pytree_dict["tensor1"])
    assert torch.equal(retrieved["tensor2"], pytree_dict["tensor2"])
    assert torch.equal(retrieved["nested"]["tensor3"], pytree_dict["nested"]["tensor3"])

    # Test with list PyTree
    sharded_storage.register_buffer("local_list", config)
    pytree_list = [torch.randn(10), torch.randn(5, 5), torch.randn(3, 3)]
    sharded_storage["local_list"] = pytree_list
    
    assert sharded_storage.buffer_states["local_list"] == BufferState.LOCAL
    retrieved_list = sharded_storage["local_list"]
    for i, tensor in enumerate(pytree_list):
        assert torch.equal(retrieved_list[i], tensor)

    # Test deletion
    sharded_storage["local_pytree"] = None
    assert sharded_storage.buffer_states["local_pytree"] == BufferState.UNINITIALIZED
    assert not sharded_storage.is_initialized("local_pytree")
    assert sharded_storage["local_pytree"] is None


def test_set_sharded_buffer(sharded_storage):
    """Tests setting a sharded buffer, which should put it in the NEEDS_SHARDING state."""
    config = BufferConfig(shard=True)
    sharded_storage.register_buffer("sharded_buffer", config)
    tensor = torch.randn(10)
    sharded_storage["sharded_buffer"] = tensor

    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.NEEDS_SHARDING
    assert torch.equal(sharded_storage["sharded_buffer"], tensor)


def test_set_sharded_buffer_pytree(sharded_storage):
    """Tests that sharded PyTree buffers are now supported."""
    config = BufferConfig(shard=True)
    sharded_storage.register_buffer("sharded_pytree", config)
    
    # PyTree structures are now supported for sharded buffers
    pytree = {
        "tensor1": torch.randn(20, 10),
        "tensor2": torch.randn(30, 15),
        "nested": {
            "tensor3": torch.randn(40, 20)
        }
    }
    # This should work now
    sharded_storage["sharded_pytree"] = pytree

    assert sharded_storage.buffer_states["sharded_pytree"] == BufferState.NEEDS_SHARDING
    retrieved = sharded_storage["sharded_pytree"]
    assert torch.equal(retrieved["tensor1"], pytree["tensor1"])
    assert torch.equal(retrieved["tensor2"], pytree["tensor2"])
    assert torch.equal(retrieved["nested"]["tensor3"], pytree["nested"]["tensor3"])


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


def test_delete_buffer_pytree(sharded_storage):
    """Tests the deletion of a PyTree buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("pytree_to_delete", config)
    pytree = {"a": torch.randn(10), "b": [torch.randn(5), torch.randn(3)]}
    sharded_storage["pytree_to_delete"] = pytree

    assert sharded_storage.is_initialized("pytree_to_delete")
    del sharded_storage["pytree_to_delete"]
    assert not sharded_storage.is_initialized("pytree_to_delete")
    assert sharded_storage.buffer_states["pytree_to_delete"] == BufferState.UNINITIALIZED


@patch("torch.distributed.tensor.DTensor.from_local")
def test_sharded_buffer_lifecycle(mock_from_local, sharded_storage, mock_state):
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
    sharded_storage.dematerialize_buffer("sharded_buffer")
    assert sharded_storage.buffer_states["sharded_buffer"] == BufferState.SHARDED
    assert "sharded_buffer" not in sharded_storage.unsharded_buffers
    mock_state.wait_for_everyone.assert_called_once()

    # 5. Delete sharded buffer
    del sharded_storage["sharded_buffer"]
    assert not sharded_storage.is_initialized("sharded_buffer")


@patch("torch.distributed.tensor.DTensor.from_local")
def test_sharded_pytree_lifecycle(mock_from_local, sharded_storage, mock_state):
    """Tests the full state transition lifecycle of a sharded PyTree buffer."""
    config = BufferConfig(shard=True, shard_dim=0)
    sharded_storage.register_buffer("sharded_pytree", config)
    
    pytree = {
        "tensor1": torch.randn(20, 10),
        "tensor2": torch.randn(30, 15),
        "nested": {
            "tensor3": torch.randn(40, 20)
        }
    }

    # Keep track of tensor IDs to mock appropriately
    tensor_ids = {
        "tensor1": id(pytree["tensor1"]),
        "tensor2": id(pytree["tensor2"]),
        "tensor3": id(pytree["nested"]["tensor3"])
    }

    # Create mock DTensors for each tensor
    mock_dtensor1 = MagicMock()
    mock_dtensor2 = MagicMock()
    mock_dtensor3 = MagicMock()
    
    # Mock redistributed tensors
    mock_sharded1 = MagicMock()
    mock_sharded2 = MagicMock()
    mock_sharded3 = MagicMock()
    
    # Mock local tensors after to_local
    mock_local1 = MagicMock()
    mock_local2 = MagicMock()
    mock_local3 = MagicMock()

    # Setup redistribution
    mock_dtensor1.redistribute.return_value = mock_sharded1
    mock_dtensor2.redistribute.return_value = mock_sharded2
    mock_dtensor3.redistribute.return_value = mock_sharded3
    
    mock_sharded1.redistribute.return_value = mock_dtensor1
    mock_sharded2.redistribute.return_value = mock_dtensor2
    mock_sharded3.redistribute.return_value = mock_dtensor3
    
    mock_dtensor1.to_local.return_value = mock_local1
    mock_dtensor2.to_local.return_value = mock_local2
    mock_dtensor3.to_local.return_value = mock_local3

    # Map tensor IDs to their mocks
    tensor_to_mock = {}
    
    def from_local_side_effect(tensor, **kwargs):
        tid = id(tensor)
        if tid == tensor_ids["tensor1"]:
            return mock_dtensor1
        elif tid == tensor_ids["tensor2"]:
            return mock_dtensor2
        elif tid == tensor_ids["tensor3"]:
            return mock_dtensor3
        return MagicMock()

    mock_from_local.side_effect = from_local_side_effect

    # 1. Set PyTree -> NEEDS_SHARDING
    sharded_storage["sharded_pytree"] = pytree
    assert sharded_storage.buffer_states["sharded_pytree"] == BufferState.NEEDS_SHARDING

    # 2. Dematerialize -> SHARDED
    sharded_storage.dematerialize_buffer("sharded_pytree")
    assert sharded_storage.buffer_states["sharded_pytree"] == BufferState.SHARDED
    assert "sharded_pytree" in sharded_storage.sharded_buffers
    
    # Verify from_local was called for each tensor
    assert mock_from_local.call_count == 3
    
    # Verify redistribute was called to shard each tensor
    mock_dtensor1.redistribute.assert_called_with(
        sharded_storage.mesh, placements=[Shard(config.shard_dim)]
    )
    mock_dtensor2.redistribute.assert_called_with(
        sharded_storage.mesh, placements=[Shard(config.shard_dim)]
    )
    mock_dtensor3.redistribute.assert_called_with(
        sharded_storage.mesh, placements=[Shard(config.shard_dim)]
    )
    
    # For PyTrees, the error will be AttributeError since dict doesn't have .placements
    with pytest.raises((ValueError, AttributeError)):
        _ = sharded_storage["sharded_pytree"]

    # 3. Materialize -> REPLICATED
    sharded_storage.materialize_buffer("sharded_pytree")
    assert sharded_storage.buffer_states["sharded_pytree"] == BufferState.REPLICATED
    assert "sharded_pytree" in sharded_storage.unsharded_buffers
    
    # Verify redistribute was called to replicate each tensor
    mock_sharded1.redistribute.assert_called_with(
        sharded_storage.mesh, placements=[Replicate()]
    )
    mock_sharded2.redistribute.assert_called_with(
        sharded_storage.mesh, placements=[Replicate()]
    )
    mock_sharded3.redistribute.assert_called_with(
        sharded_storage.mesh, placements=[Replicate()]
    )

    # 4. Delete sharded PyTree buffer
    del sharded_storage["sharded_pytree"]
    assert not sharded_storage.is_initialized("sharded_pytree")


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


def test_accumulate_local_buffer(sharded_storage):
    """Tests accumulation on a local buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("local_acc", config)
    
    # Initial tensor
    initial = torch.ones(5, 5)
    sharded_storage["local_acc"] = initial
    
    # Accumulate
    to_add = torch.ones(5, 5) * 2
    sharded_storage.accumulate("local_acc", to_add)
    
    result = sharded_storage["local_acc"]
    expected = torch.ones(5, 5) * 3
    assert torch.allclose(result, expected)


def test_accumulate_local_pytree(sharded_storage):
    """Tests accumulation on a local PyTree buffer."""
    config = BufferConfig(shard=False)
    sharded_storage.register_buffer("local_pytree_acc", config)
    
    # Initial PyTree
    initial = {
        "tensor1": torch.ones(3, 3),
        "tensor2": torch.ones(4, 4) * 2,
        "nested": {
            "tensor3": torch.ones(2, 2) * 3
        }
    }
    sharded_storage["local_pytree_acc"] = initial
    
    # PyTree to accumulate
    to_add = {
        "tensor1": torch.ones(3, 3) * 0.5,
        "tensor2": torch.ones(4, 4),
        "nested": {
            "tensor3": torch.ones(2, 2) * 2
        }
    }
    sharded_storage.accumulate("local_pytree_acc", to_add)
    
    result = sharded_storage["local_pytree_acc"]
    assert torch.allclose(result["tensor1"], torch.ones(3, 3) * 1.5)
    assert torch.allclose(result["tensor2"], torch.ones(4, 4) * 3)
    assert torch.allclose(result["nested"]["tensor3"], torch.ones(2, 2) * 5)


@patch("torch.distributed.tensor.DTensor.from_local")
def test_accumulate_sharded_buffer(mock_from_local, sharded_storage):
    """Tests accumulation on a sharded buffer."""
    config = BufferConfig(shard=True, shard_dim=0)
    sharded_storage.register_buffer("sharded_acc", config)
    
    # Mock sharded tensor
    mock_sharded = MagicMock()
    mock_sharded.device_mesh = sharded_storage.mesh
    sharded_storage.sharded_buffers["sharded_acc"] = mock_sharded
    sharded_storage.buffer_states["sharded_acc"] = BufferState.SHARDED
    
    # Mock DTensor creation and redistribution
    mock_partial_tensor = MagicMock()
    mock_from_local.return_value = mock_partial_tensor
    
    mock_sharded_value = MagicMock()
    mock_partial_tensor.redistribute.return_value = mock_sharded_value
    
    # Value to accumulate
    to_add = torch.ones(10, 10)
    sharded_storage.accumulate("sharded_acc", to_add)
    
    # Verify DTensor creation with Partial placement
    mock_from_local.assert_called_once_with(
        to_add,
        device_mesh=sharded_storage.mesh,
        placements=[Partial(reduce_op='sum')]
    )
    
    # Verify redistribution to Shard placement
    mock_partial_tensor.redistribute.assert_called_once_with(
        sharded_storage.mesh,
        placements=[Shard(config.shard_dim)]
    )
    
    # Verify add_ was called on the sharded tensor
    mock_sharded.add_.assert_called_once_with(mock_sharded_value)


@patch("torch.distributed.tensor.DTensor.from_local")
def test_accumulate_sharded_pytree(mock_from_local, sharded_storage):
    """Tests accumulation on a sharded PyTree buffer."""
    config = BufferConfig(shard=True, shard_dim=0)
    sharded_storage.register_buffer("sharded_pytree_acc", config)
    
    # Mock sharded PyTree
    mock_sharded_tensor1 = MagicMock()
    mock_sharded_tensor2 = MagicMock()
    mock_sharded_tensor3 = MagicMock()
    
    mock_sharded_pytree = {
        "tensor1": mock_sharded_tensor1,
        "tensor2": mock_sharded_tensor2,
        "nested": {"tensor3": mock_sharded_tensor3}
    }
    sharded_storage.sharded_buffers["sharded_pytree_acc"] = mock_sharded_pytree
    sharded_storage.buffer_states["sharded_pytree_acc"] = BufferState.SHARDED
    
    # Mock partial tensors that will be created
    mock_partial1 = MagicMock()
    mock_partial2 = MagicMock()
    mock_partial3 = MagicMock()
    
    # Mock sharded values after redistribution
    mock_sharded_value1 = MagicMock()
    mock_sharded_value2 = MagicMock()
    mock_sharded_value3 = MagicMock()
    
    # Setup redistribution
    mock_partial1.redistribute.return_value = mock_sharded_value1
    mock_partial2.redistribute.return_value = mock_sharded_value2
    mock_partial3.redistribute.return_value = mock_sharded_value3
    
    # Counter to track which tensor we're processing
    call_count = 0
    tensor_to_partial = {}
    
    # PyTree to accumulate
    to_add = {
        "tensor1": torch.ones(10, 10),
        "tensor2": torch.ones(20, 20),
        "nested": {"tensor3": torch.ones(30, 30)}
    }
    
    # Map tensors to their mocks before calling accumulate
    tensor_to_partial[id(to_add["tensor1"])] = mock_partial1
    tensor_to_partial[id(to_add["tensor2"])] = mock_partial2
    tensor_to_partial[id(to_add["nested"]["tensor3"])] = mock_partial3
    
    def from_local_side_effect(tensor, **kwargs):
        return tensor_to_partial.get(id(tensor), MagicMock())
    
    mock_from_local.side_effect = from_local_side_effect
    
    sharded_storage.accumulate("sharded_pytree_acc", to_add)
    
    # Verify DTensor.from_local was called for each tensor
    assert mock_from_local.call_count == 3
    
    # Verify redistribution was called on each partial tensor
    mock_partial1.redistribute.assert_called_once_with(
        sharded_storage.mesh,
        placements=[Shard(config.shard_dim)]
    )
    mock_partial2.redistribute.assert_called_once_with(
        sharded_storage.mesh,
        placements=[Shard(config.shard_dim)]
    )
    mock_partial3.redistribute.assert_called_once_with(
        sharded_storage.mesh,
        placements=[Shard(config.shard_dim)]
    )
    
    # Verify add_ was called on each tensor in the sharded PyTree
    mock_sharded_tensor1.add_.assert_called_once_with(mock_sharded_value1)
    mock_sharded_tensor2.add_.assert_called_once_with(mock_sharded_value2)
    mock_sharded_tensor3.add_.assert_called_once_with(mock_sharded_value3) 