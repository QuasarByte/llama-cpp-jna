package com.quasarbyte.llama.cpp.jna.binding.cuda;

import com.quasarbyte.llama.cpp.jna.library.declaration.cuda.GgmlCudaLibrary;
import com.quasarbyte.llama.cpp.jna.model.cuda.CudaDevice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class GgmlCudaDeviceReaderBindingImpl implements GgmlCudaDeviceReaderBinding {

    private static final Logger logger = LoggerFactory.getLogger(GgmlCudaDeviceReaderBindingImpl.class);

    private final GgmlCudaLibrary ggmlCudaLibrary;

    public GgmlCudaDeviceReaderBindingImpl(GgmlCudaLibrary ggmlCudaLibrary) {
        this.ggmlCudaLibrary = ggmlCudaLibrary;
    }

    @Override
    public List<CudaDevice> findAll() {

        final List<CudaDevice> devices = new ArrayList<>();

        int cudaDeviceCount = ggmlCudaLibrary.ggml_backend_cuda_get_device_count();

        if (cudaDeviceCount > 0) {

            for (int deviceId = 0; deviceId < cudaDeviceCount; deviceId++) {
                byte[] descBuffer = new byte[512];
                ggmlCudaLibrary.ggml_backend_cuda_get_device_description(deviceId, descBuffer, descBuffer.length);
                String description = new String(descBuffer).trim();
                // Remove null bytes
                int nullIndex = description.indexOf('\0');
                if (nullIndex >= 0) {
                    description = description.substring(0, nullIndex);
                }

                devices.add(new CudaDevice().setId(deviceId).setDescription(description));
            }
        }

        return devices;
    }

}
