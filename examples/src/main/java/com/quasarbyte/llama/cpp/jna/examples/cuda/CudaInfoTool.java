package com.quasarbyte.llama.cpp.jna.examples.cuda;

import com.quasarbyte.llama.cpp.jna.library.declaration.cuda.GgmlCudaLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.cuda.GgmlCudaLibraryFactory;
import com.quasarbyte.llama.cpp.jna.model.cuda.CudaDevice;
import com.quasarbyte.llama.cpp.jna.binding.cuda.GgmlCudaDeviceReaderBinding;
import com.quasarbyte.llama.cpp.jna.binding.cuda.GgmlCudaDeviceReaderBindingFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Lightweight utility that prints basic CUDA device information.
 */
public final class CudaInfoTool {

    private static final Logger logger = LoggerFactory.getLogger(CudaInfoTool.class);

    private CudaInfoTool() {
        // Utility class
    }

    public static void main(String[] args) {
        GgmlCudaLibrary cudaLibrary = new GgmlCudaLibraryFactory().getInstance();
        GgmlCudaDeviceReaderBinding deviceService = new GgmlCudaDeviceReaderBindingFactory(cudaLibrary).create();

        try {
            List<CudaDevice> devices = deviceService.findAll();
            if (devices.isEmpty()) {
                System.out.println("No CUDA devices detected.");
                return;
            }

            System.out.println("Detected CUDA devices:");
            devices.forEach(device -> {
                Integer id = device.getId();
                String description = device.getDescription();
                System.out.printf("  #%d - %s%n",
                        id != null ? id : -1,
                        description != null ? description : "Unknown device");
            });
        } catch (Throwable t) {
            String message = t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName();
            logger.error("Unable to enumerate CUDA devices: {}", message, t);
            System.out.println("CUDA is not available on this system (" + message + ").");
            System.out.println("Ensure the CUDA runtime libraries are installed and on the PATH.");
        }
    }
}
