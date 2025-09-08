package com.quasarbyte.llama.cpp.jna.binding.llama.system;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.sun.jna.Pointer;

import java.util.Objects;

public class LlamaSystemBindingImpl implements LlamaSystemBinding {

    private final LlamaLibrary llamaLibrary;

    public LlamaSystemBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public void initBackend() {
        llamaLibrary.llama_backend_init();
    }

    @Override
    public void freeBackend() {
        llamaLibrary.llama_backend_free();
    }

    @Override
    public void initNuma(int numa) {
        llamaLibrary.llama_numa_init(numa);
    }

    @Override
    public boolean supportsMmap() {
        return llamaLibrary.llama_supports_mmap();
    }

    @Override
    public boolean supportsMlock() {
        return llamaLibrary.llama_supports_mlock();
    }

    @Override
    public boolean supportsGpuOffload() {
        return llamaLibrary.llama_supports_gpu_offload();
    }

    @Override
    public boolean supportsRpc() {
        return llamaLibrary.llama_supports_rpc();
    }

    @Override
    public long getMaxDevices() {
        return llamaLibrary.llama_max_devices();
    }

    @Override
    public long getMaxParallelSequences() {
        return llamaLibrary.llama_max_parallel_sequences();
    }

    @Override
    public long getCurrentTimeUs() {
        return llamaLibrary.llama_time_us();
    }

    @Override
    public void attachThreadpool(LlamaContext context, LlamaThreadPool threadPool, LlamaThreadPoolBatch threadPoolBatchPointer) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        Objects.requireNonNull(threadPool);
        Objects.requireNonNull(threadPool.getLlamaThreadPoolNative());
        Objects.requireNonNull(threadPoolBatchPointer);
        Objects.requireNonNull(threadPoolBatchPointer.getLlamaThreadPoolNative());
        llamaLibrary.llama_attach_threadpool(context.getContextPointer(), threadPool.getLlamaThreadPoolNative(), threadPoolBatchPointer.getLlamaThreadPoolNative());
    }

    @Override
    public void detachThreadpool(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        llamaLibrary.llama_detach_threadpool(context.getContextPointer());
    }

    @Override
    public String getSystemInfo() {
        return llamaLibrary.llama_print_system_info();
    }

    @Override
    public void setLogCallback(LlamaLibrary.LlamaLogCallback logCallback, Pointer userDataPointer) {
        llamaLibrary.llama_log_set(logCallback, userDataPointer);
    }

    @Override
    public String getFlashAttnTypeName(int flashAttnType) {
        return llamaLibrary.llama_flash_attn_type_name(flashAttnType);
    }
}