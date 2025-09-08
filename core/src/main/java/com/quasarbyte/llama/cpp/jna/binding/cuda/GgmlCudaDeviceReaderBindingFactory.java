package com.quasarbyte.llama.cpp.jna.binding.cuda;

import com.quasarbyte.llama.cpp.jna.library.declaration.cuda.GgmlCudaLibrary;

public class GgmlCudaDeviceReaderBindingFactory {

    private final GgmlCudaLibrary ggmlCudaLibrary;

    public GgmlCudaDeviceReaderBindingFactory(GgmlCudaLibrary ggmlCudaLibrary) {
        this.ggmlCudaLibrary = ggmlCudaLibrary;
    }

    public GgmlCudaDeviceReaderBinding create() {
        return new GgmlCudaDeviceReaderBindingImpl(ggmlCudaLibrary);
    }

}
