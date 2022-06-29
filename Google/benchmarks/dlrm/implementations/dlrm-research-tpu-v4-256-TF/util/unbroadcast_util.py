"""A shared utility function used in various configs."""

import jax


def _unbroadcast(x: jax.pxla.ShardedDeviceArray) -> jax.pxla.ShardedDeviceArray:
  """Assuming `x` is replicated along its leading axis, remove that axis."""
  # Unbroadcast is a hack to take the output of a pmap with out_axes=0 and turn
  # it into the input of a pmap with in_axes=None. This is necessary because we
  # don't have out_axes=None in pmap, so the output arrays of the training step
  # function all still end up with an extra leading logical axis of size
  # `num_local_devices`.
  sharding_spec = x.sharding_spec
  # The leading logical axis should be sharded like the result of a pmap with
  # out_axes=0.
  assert sharding_spec.sharding[0] == jax.pxla.Unstacked(x.shape[0])
  # Remove that leading logical axis and its corresponding sharding.
  aval = jax.abstract_arrays.ShapedArray(x.shape[1:], x.dtype)
  sharding = sharding_spec.sharding[1:]

  # Replace the mesh mapping entry that pointed to that axis with Replicated,
  # and decrement the other entries.
  def replace_mesh_mapping(mm):
    if isinstance(mm, jax.pxla.ShardedAxis):
      if mm.axis == 0:
        return jax.pxla.Replicated(x.shape[0])
      return jax.pxla.ShardedAxis(mm.axis - 1)
    return mm

  mesh_mapping = map(replace_mesh_mapping, sharding_spec.mesh_mapping)
  sharding_spec = jax.pxla.ShardingSpec(sharding, mesh_mapping)
  constructor = getattr(jax.pxla, "make_sharded_device_array",
                        jax.pxla.ShardedDeviceArray)
  return constructor(aval, sharding_spec, x.device_buffers)


def unbroadcast(tree):
  """Assuming `tree` is replicated along its leading axis, remove that axis."""
  return jax.tree_map(_unbroadcast, tree)
