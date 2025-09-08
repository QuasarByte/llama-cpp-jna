package com.quasarbyte.llama.cpp.jna.binding.llama.context;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContextParamsNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.sun.jna.Pointer;

public interface LlamaContextBinding {
    // Context Creation & Management
    LlamaContext create(LlamaModel model);

    LlamaContext create(LlamaModel model, LlamaContextParamsNative params);

    LlamaContext create(LlamaModel model, int nCtx, int nBatch, boolean noPerf);

    long getContextSize(LlamaContext context);

    long getMicroBatchSize(LlamaContext context);

    long getMaxNumberOfSequences(LlamaContext context);

    void printPerformance(LlamaContext context);

    void freeContext(LlamaContext context);

    // Context Configuration Functions
    int getPoolingType(LlamaContext context);

    void setThreadCount(LlamaContext context, int nThreads, int nThreadsBatch);

    int getThreadCount(LlamaContext context);

    int getBatchThreadCount(LlamaContext context);

    void setEmbeddings(LlamaContext context, boolean embeddings);

    void setCausalAttention(LlamaContext context, boolean causalAttention);

    void setWarmup(LlamaContext context, boolean warmup);

    void setAbortCallback(LlamaContext context, Pointer abortCallback, Pointer abortCallbackData);

    void synchronize(LlamaContext context);
}
