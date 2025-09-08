package com.quasarbyte.llama.cpp.jna.binding.llama.state;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContextNative;
import com.sun.jna.Pointer;

import java.nio.file.Path;

public interface LlamaStateBinding {

    long getStateSize(LlamaContextNative ctx);

    long getStateData(LlamaContextNative ctx, Pointer dst, long size);

    long setStateData(LlamaContextNative ctx, Pointer src, long size);

    boolean loadStateFromFile(LlamaContextNative ctx, Path pathSession, Pointer tokensOut, long nTokenCapacity, Pointer nTokenCountOut);

    boolean saveStateToFile(LlamaContextNative ctx, Path pathSession, Pointer tokens, long nTokenCount);

    long getSequenceStateSize(LlamaContextNative ctx, int seqId);

    long setSequenceStateData(LlamaContextNative ctx, Pointer src, long size, int destSeqId);

    long saveSequenceStateToFile(LlamaContextNative ctx, Path filepath, int seqId, Pointer tokens, long nTokenCount);

    long loadSequenceStateFromFile(LlamaContextNative ctx, Path filepath, int destSeqId, Pointer tokensOut, long nTokenCapacity, Pointer nTokenCountOut);

    long getSequenceStateSizeExt(LlamaContextNative ctx, int seqId, int flags);

    long getSequenceStateDataExt(LlamaContextNative ctx, Pointer dst, long size, int seqId, int flags);

    long setSequenceStateDataExt(LlamaContextNative ctx, Pointer src, long size, int destSeqId, int flags);
}