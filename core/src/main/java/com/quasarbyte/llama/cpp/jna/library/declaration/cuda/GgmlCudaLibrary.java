package com.quasarbyte.llama.cpp.jna.library.declaration.cuda;

import com.quasarbyte.llama.cpp.jna.library.declaration.LlamaCppBase;
import com.sun.jna.Pointer;

// Interface for CUDA backend functions
public interface GgmlCudaLibrary extends LlamaCppBase<GgmlCudaLibrary> {

    // Backend initialization functions from ggml-cuda
    Pointer ggml_backend_init(); // Generic backend init

    Pointer ggml_backend_cuda_init(int device); // CUDA specific backend init

    Pointer ggml_backend_cuda_reg(); // Get CUDA backend registry

    // Device management functions
    int ggml_backend_cuda_get_device_count(); // Get number of CUDA devices

    void ggml_backend_cuda_get_device_description(int device, byte[] description, long description_size); // Get device description

    void ggml_backend_cuda_get_device_memory(int device, Pointer free, Pointer total); // Get device memory info

    // Buffer type functions
    Pointer ggml_backend_cuda_buffer_type(int device); // Get CUDA buffer type for device

    Pointer ggml_backend_cuda_host_buffer_type(); // Get CUDA host buffer type

    Pointer ggml_backend_cuda_split_buffer_type(int main_device, Pointer tensor_split); // Get split buffer type

    // Host buffer registration functions
    boolean ggml_backend_cuda_register_host_buffer(Pointer buffer, long size); // Register host buffer

    void ggml_backend_cuda_unregister_host_buffer(Pointer buffer); // Unregister host buffer
}
