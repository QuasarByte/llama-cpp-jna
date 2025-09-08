package com.quasarbyte.llama.cpp.jna.binding.llama.memory;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaMemoryManager;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaMemoryManagerNative;

import java.util.Objects;

public class LlamaMemoryBindingImpl implements LlamaMemoryBinding {

    private final LlamaLibrary llamaLibrary;

    public LlamaMemoryBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public LlamaMemoryManager getMemory(LlamaContext context) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(context.getContextPointer(), "context pointer");

        LlamaMemoryManagerNative pointer = llamaLibrary.llama_get_memory(context.getContextPointer());
        if (pointer == null || pointer.getPointer() == null) {
            throw new LlamaCppJnaException("Memory manager pointer is null");
        }

        return new LlamaMemoryManager(pointer);
    }

    @Override
    public void clear(LlamaMemoryManager memoryManager, boolean data) {
        llamaLibrary.llama_memory_clear(requirePointer(memoryManager), data);
    }

    @Override
    public boolean sequenceRemove(LlamaMemoryManager memoryManager, int seqId, int p0, int p1) {
        return llamaLibrary.llama_memory_seq_rm(requirePointer(memoryManager), seqId, p0, p1);
    }

    @Override
    public void sequenceCopy(LlamaMemoryManager memoryManager, int seqIdSrc, int seqIdDst, int p0, int p1) {
        llamaLibrary.llama_memory_seq_cp(requirePointer(memoryManager), seqIdSrc, seqIdDst, p0, p1);
    }

    @Override
    public void sequenceKeep(LlamaMemoryManager memoryManager, int seqId) {
        llamaLibrary.llama_memory_seq_keep(requirePointer(memoryManager), seqId);
    }

    @Override
    public void sequenceAdd(LlamaMemoryManager memoryManager, int seqId, int p0, int p1, int delta) {
        llamaLibrary.llama_memory_seq_add(requirePointer(memoryManager), seqId, p0, p1, delta);
    }

    @Override
    public void sequenceDivide(LlamaMemoryManager memoryManager, int seqId, int p0, int p1, int d) {
        llamaLibrary.llama_memory_seq_div(requirePointer(memoryManager), seqId, p0, p1, d);
    }

    @Override
    public int sequencePositionMin(LlamaMemoryManager memoryManager, int seqId) {
        return llamaLibrary.llama_memory_seq_pos_min(requirePointer(memoryManager), seqId);
    }

    @Override
    public int sequencePositionMax(LlamaMemoryManager memoryManager, int seqId) {
        return llamaLibrary.llama_memory_seq_pos_max(requirePointer(memoryManager), seqId);
    }

    @Override
    public boolean canShift(LlamaMemoryManager memoryManager) {
        return llamaLibrary.llama_memory_can_shift(requirePointer(memoryManager));
    }

    private LlamaMemoryManagerNative requirePointer(LlamaMemoryManager manager) {
        Objects.requireNonNull(manager, "memory manager");
        LlamaMemoryManagerNative pointer = manager.getPointer();
        if (pointer == null || pointer.getPointer() == null) {
            throw new IllegalArgumentException("Memory manager pointer is null");
        }
        return pointer;
    }
}
