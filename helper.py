# Import necessary modules for garbage collection
import gc
import torch
# Function to clear CUDA cache and perform garbage collection
def clear_cuda_cache():
    """
    Clears CUDA cache and performs garbage collection to free up GPU memory.
    This helps prevent out-of-memory errors during training.
    """
    # Clear CUDA cache if torch is using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection to free up memory
    gc.collect()

def print_cuda_memory_stats():
    """
    Prints basic CUDA memory statistics for all available GPUs.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} - {torch.cuda.get_device_name(i)}:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")

def track_tensor_memory():
    """
    Tracks tensor memory usage to identify what's consuming GPU memory.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype, obj.device, 
                      f"{obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
        except:
            pass

def detailed_memory_summary():
    """
    Provides a detailed memory summary from PyTorch.
    """
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())
