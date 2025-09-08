package com.quasarbyte.llama.cpp.jna.binding.llama.state;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallLongResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContextNative;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Objects;

public class LlamaStateBindingImpl implements LlamaStateBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaStateBindingImpl.class);

    private final LlamaLibrary llamaLibrary;

    public LlamaStateBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public long getStateSize(LlamaContextNative ctx) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());
        return llamaLibrary.llama_state_get_size(ctx);
    }

    @Override
    public long getStateData(LlamaContextNative ctx, Pointer dst, long size) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());

        long length = llamaLibrary.llama_state_get_data(ctx, dst, size);

        if (length < 0) {
            logger.error("Failed to copy state data to a buffer, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to copy state data to a buffer, length: %d", length));
        }

        return length;
    }

    @Override
    public long setStateData(LlamaContextNative ctx, Pointer src, long size) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());

        long length = llamaLibrary.llama_state_set_data(ctx, src, size);

        if (length < 0) {
            logger.error("Failed to load state data from a buffer, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to load state data from a buffer, length: %d", length));
        }

        return llamaLibrary.llama_state_set_data(ctx, src, size);
    }

    @Override
    public boolean loadStateFromFile(LlamaContextNative ctx, Path pathSession, Pointer tokensOut, long nTokenCapacity, Pointer nTokenCountOut) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());
        Objects.requireNonNull(pathSession);

        Path absolutePath = pathSession.toAbsolutePath();

        if (!absolutePath.toFile().exists()) {
            logger.error("Path session does not exist: {}", absolutePath);
            throw new LlamaCppJnaException(String.format("Path session does not exist: %s", absolutePath));
        }

        return llamaLibrary.llama_state_load_file(ctx, pathSession.toAbsolutePath().toString(), tokensOut, nTokenCapacity, nTokenCountOut);
    }

    @Override
    public boolean saveStateToFile(LlamaContextNative ctx, Path pathSession, Pointer tokens, long nTokenCount) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());
        Objects.requireNonNull(pathSession);
        return llamaLibrary.llama_state_save_file(ctx, pathSession.toString(), tokens, nTokenCount);
    }

    @Override
    public long getSequenceStateSize(LlamaContextNative ctx, int seqId) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());
        return llamaLibrary.llama_state_seq_get_size(ctx, seqId);
    }

    @Override
    public long setSequenceStateData(LlamaContextNative ctx, Pointer src, long size, int destSeqId) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());

        long length = llamaLibrary.llama_state_seq_set_data(ctx, src, size, destSeqId);

        if (length < 0) {
            logger.error("Failed to load sequence state data from a buffer, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to load sequence state data from a buffer, length: %d", length));
        }

        return length;
    }

    @Override
    public long saveSequenceStateToFile(LlamaContextNative ctx, Path filepath, int seqId, Pointer tokens, long nTokenCount) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());
        Objects.requireNonNull(filepath);

        long length = llamaLibrary.llama_state_seq_save_file(ctx, filepath.toString(), seqId, tokens, nTokenCount);

        if (length < 0) {
            logger.error("Failed to Save sequence state to a file, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to Save sequence state to a file, length: %d", length));
        }

        return length;
    }

    @Override
    public long loadSequenceStateFromFile(LlamaContextNative ctx, Path filepath, int destSeqId, Pointer tokensOut, long nTokenCapacity, Pointer nTokenCountOut) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());
        Objects.requireNonNull(filepath);

        long length = llamaLibrary.llama_state_seq_load_file(ctx, filepath.toString(), destSeqId, tokensOut, nTokenCapacity, nTokenCountOut);

        if (length < 0) {
            logger.error("Failed to load sequence state from a file, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to load sequence state from a file, length: %d", length));
        }

        return length;
    }

    @Override
    public long getSequenceStateSizeExt(LlamaContextNative ctx, int seqId, int flags) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());

        long length = llamaLibrary.llama_state_seq_get_size_ext(ctx, seqId, flags);

        if (length < 0) {
            logger.error("Failed to get the size in bytes of extended sequence state data, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to get the size in bytes of extended sequence state data, length: %d", length));
        }

        return length;
    }

    @Override
    public long getSequenceStateDataExt(LlamaContextNative ctx, Pointer dst, long size, int seqId, int flags) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());

        long length = llamaLibrary.llama_state_seq_get_data_ext(ctx, dst, size, seqId, flags);

        if (length < 0) {
            logger.error("Failed to copy extended sequence state data to a buffer, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to copy extended sequence state data to a buffer, length: %d", length));
        }

        return length;
    }

    @Override
    public long setSequenceStateDataExt(LlamaContextNative ctx, Pointer src, long size, int destSeqId, int flags) {
        Objects.requireNonNull(ctx);
        Objects.requireNonNull(ctx.getPointer());

        long length = llamaLibrary.llama_state_seq_set_data_ext(ctx, src, size, destSeqId, flags);

        if (length < 0) {
            logger.error("Failed to load extended sequence state data from a buffer, length: {}", length);
            throw new LlamaFunctionCallLongResultException(length, String.format("Failed to load extended sequence state data from a buffer, length: %d", length));
        }

        return length;
    }
}