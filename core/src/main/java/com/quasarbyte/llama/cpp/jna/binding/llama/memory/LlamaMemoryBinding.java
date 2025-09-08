package com.quasarbyte.llama.cpp.jna.binding.llama.memory;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaMemoryManager;

public interface LlamaMemoryBinding {

    LlamaMemoryManager getMemory(LlamaContext context);

    void clear(LlamaMemoryManager memoryPointer, boolean data);

    boolean sequenceRemove(LlamaMemoryManager memoryPointer, int seqId, int p0, int p1);

    void sequenceCopy(LlamaMemoryManager memoryPointer, int seqIdSrc, int seqIdDst, int p0, int p1);

    void sequenceKeep(LlamaMemoryManager memoryPointer, int seqId);

    void sequenceAdd(LlamaMemoryManager memoryPointer, int seqId, int p0, int p1, int delta);

    void sequenceDivide(LlamaMemoryManager memoryPointer, int seqId, int p0, int p1, int d);

    int sequencePositionMin(LlamaMemoryManager memoryPointer, int seqId);

    int sequencePositionMax(LlamaMemoryManager memoryPointer, int seqId);

    boolean canShift(LlamaMemoryManager memoryPointer);
}
