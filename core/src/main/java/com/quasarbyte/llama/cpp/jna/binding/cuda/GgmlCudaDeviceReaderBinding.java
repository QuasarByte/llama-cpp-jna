package com.quasarbyte.llama.cpp.jna.binding.cuda;

import com.quasarbyte.llama.cpp.jna.model.cuda.CudaDevice;

import java.util.List;

public interface GgmlCudaDeviceReaderBinding {
    List<CudaDevice> findAll();
}
