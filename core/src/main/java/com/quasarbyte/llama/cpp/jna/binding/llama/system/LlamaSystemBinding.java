package com.quasarbyte.llama.cpp.jna.binding.llama.system;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaThreadPool;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaThreadPoolBatch;
import com.sun.jna.Pointer;

public interface LlamaSystemBinding {

    void initBackend();

    void freeBackend();

    void initNuma(int numa);

    boolean supportsMmap();

    boolean supportsMlock();

    boolean supportsGpuOffload();

    boolean supportsRpc();

    long getMaxDevices();

    long getMaxParallelSequences();

    long getCurrentTimeUs();

    void attachThreadpool(LlamaContext context, LlamaThreadPool threadPool, LlamaThreadPoolBatch threadPoolBatch);

    void detachThreadpool(LlamaContext context);

    String getSystemInfo();

    void setLogCallback(LlamaLibrary.LlamaLogCallback logCallback, Pointer userDataPointer);

    String getFlashAttnTypeName(int flashAttnType);
}