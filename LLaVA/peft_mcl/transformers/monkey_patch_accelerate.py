import accelerate
from accelerate.utils.modeling import set_module_tensor_to_device as original_set_module_tensor_to_device

def patched_set_module_tensor_to_device(module, tensor_name, device, value=None, **kwargs):
    # Check if the module has a `max_source_positions_pretrained` attribute and we're setting the "weight".
    if hasattr(module, "audio_tower") and "embed_positions.weight" in tensor_name:
        expected_rows = module.config.audio_config.max_source_positions
        if value is not None and value.size(0) != expected_rows:
            # Slice the loaded weight to the expected shape.
            value = value[:expected_rows, :]
    
    return original_set_module_tensor_to_device(module, tensor_name, device, value=value, **kwargs)

# Monkey-patch the Accelerate function.
accelerate.utils.modeling.set_module_tensor_to_device = patched_set_module_tensor_to_device