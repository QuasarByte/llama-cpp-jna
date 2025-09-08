package com.quasarbyte.llama.cpp.jna.binding.llama.processing;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaBatch;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/**
 * Implementation of LlamaProcessingBinding for inference operations.
 */
public class LlamaProcessingBindingImpl implements LlamaProcessingBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaProcessingBindingImpl.class);

    private final LlamaLibrary llamaLibrary;

    public LlamaProcessingBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public int encodeBatch(LlamaContext context, LlamaBatch batch) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        Objects.requireNonNull(batch);
        Objects.requireNonNull(batch.getNativeBatch());
        // Note: tokensMemory can be null for batches created with llama_batch_init

        try {

            int result = llamaLibrary.llama_encode(context.getContextPointer(), batch.getNativeBatch());

            if (result < 0) {
                logger.error("Failed to encode batch, result: {}", result);
                throw new LlamaFunctionCallIntResultException(result, String.format("Failed to encode batch, resultFailed to encode batch, result: %d", result));
            }

            if (result > 0) {
                logger.warn("The batch has been encoded with warning, result: {}", result);
            }

            return result;
        } catch (Exception e) {
            logger.error("Failed to encodeBatch batch, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to encodeBatch batch, error: %s", e.getMessage()), e);
        }
    }

    @Override
    public int decodeBatch(LlamaContext context, LlamaBatch batch) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        Objects.requireNonNull(batch);
        Objects.requireNonNull(batch.getNativeBatch());
        // Note: tokensMemory can be null for batches created with llama_batch_init

        try {

            int result = llamaLibrary.llama_decode(context.getContextPointer(), batch.getNativeBatch());

            if (result < 0) {
                logger.error("Failed to decoded batch, result: {}", result);
                throw new LlamaFunctionCallIntResultException(result, String.format("Failed to encode batch, resultFailed to encode batch, result: %d", result));
            }

            if (result > 0) {
                logger.warn("The batch has been decoded with warning, result: {}", result);
            }

            return result;
        } catch (Exception e) {
            logger.error("Failed to decode batch, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to decode batch, error: %s", e.getMessage()), e);
        }
    }

    @Override
    public boolean isSuccess(int result) {
        return result == 0;
    }

    @Override
    public boolean hasWarning(int result) {
        return result > 0;
    }

    @Override
    public boolean hasError(int result) {
        return result < 0;
    }

    @Override
    public String getResultDescription(int result) {
        if (result == 0) {
            return "Success";
        } else if (result > 0) {
            return "Warning (code: " + result + ")";
        } else {
            return "Error (code: " + result + ")";
        }
    }
}