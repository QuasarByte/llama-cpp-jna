package com.quasarbyte.llama.cpp.jna.binding.llama.batch;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaBatch;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaBatchNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.sun.jna.Memory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class LlamaBatchBindingImpl implements LlamaBatchBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaBatchBindingImpl.class);
    private final LlamaLibrary llamaLibrary;

    public LlamaBatchBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public LlamaBatch createBatch(int[] tokens) {
        Objects.requireNonNull(tokens);

        logger.debug("Creating batch with {} tokens", tokens.length);

        Memory tokensMemory = null;

        try {
            // Allocate memory for tokens, int32_t = 4 bytes
            tokensMemory = new Memory(tokens.length * 4L);

            // Copy tokens to memory
            for (int i = 0; i < tokens.length; i++) {
                tokensMemory.setInt(i * 4L, tokens[i]);
            }

            // Use llama_batch_get_one which creates properly initialized batches
            // This matches the working Simple examples and avoids sequence array issues
            LlamaBatchNative batch = llamaLibrary.llama_batch_get_one(tokensMemory, tokens.length);
            if (batch == null) {
                logger.error("Failed to create batch for {} tokens", tokens.length);
                throw new LlamaCppJnaException("Failed to create batch");
            }

            // Validate batch structure to prevent memory access violations
            try {
                batch.validate();
            } catch (IllegalStateException e) {
                logger.error("Batch validation failed: {}", e.getMessage());
                throw new LlamaCppJnaException("Batch validation failed: " + e.getMessage(), e);
            }

            // Log batch field states for debugging
            logger.debug("Batch created - n_tokens: {}, token: {}, embd: {}, pos: {}, n_seq_id: {}, seq_id: {}, logits: {}",
                batch.n_tokens,
                batch.token,
                batch.embd,
                batch.pos,
                batch.n_seq_id,
                batch.seq_id,
                batch.logits);

            logger.debug("Successfully created batch with {} tokens", tokens.length);

            // Return batch - memory must be managed by Java since llama_batch_get_one
            // creates a batch that references external memory, not malloc'd memory
            return new LlamaBatch().setNativeBatch(batch).setTokensMemory(tokensMemory);

        } catch (Exception e) {
            // Clean up memory only if batch creation failed
            if (tokensMemory != null) {
                tokensMemory.close();
            }

            logger.error("Failed to create batch with {} tokens, message: {}", tokens.length, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to create batch with %d tokens, message: '%s'", tokens.length, e.getMessage()), e);
        }
    }


    @Override
    public void freeBatch(LlamaBatch llamaBatch) {
        Objects.requireNonNull(llamaBatch);
        Objects.requireNonNull(llamaBatch.getNativeBatch());

        try {
            logger.debug("Freeing batch with {} tokens", llamaBatch.getTokenCount());

            if (llamaBatch.getTokensMemory() != null) {
                // This is a batch created with llama_batch_get_one - only free the JNA Memory
                // Do NOT call llama_batch_free as it would try to free() memory allocated with JNA
                logger.debug("Freeing JNA Memory for get_one batch");
                llamaBatch.getTokensMemory().close();
            } else {
                // This is a batch created with llama_batch_init - use llama_batch_free
                // as it allocates memory with malloc() that must be freed with free()
                logger.debug("Freeing init batch with llama_batch_free");
                llamaLibrary.llama_batch_free(llamaBatch.getNativeBatch());
            }

            logger.debug("Successfully freed batch");
        } catch (Exception e) {
            logger.error("Error freeing batch", e);
            throw new LlamaCppJnaException(String.format("Failed to free batch, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public LlamaBatchNative initializeBatch(int nTokens, int embeddingDimension, int nSeqMax) {
        try {
            if (nTokens <= 0) {
                throw new LlamaCppJnaException("nTokens must be positive");
            }
            if (embeddingDimension < 0) {
                throw new LlamaCppJnaException("embeddingDimension must be non-negative");
            }
            if (nSeqMax <= 0) {
                throw new LlamaCppJnaException("nSeqMax must be positive");
            }

            LlamaBatchNative batch = llamaLibrary.llama_batch_init(nTokens, embeddingDimension, nSeqMax);
            if (batch == null) {
                throw new LlamaCppJnaException("Failed to initialize batch");
            }

            logger.debug("Initialized batch: nTokens={}, embeddingDim={}, nSeqMax={}",
                    nTokens, embeddingDimension, nSeqMax);
            return batch;

        } catch (Exception e) {
            logger.error("Error initializing batch: nTokens={}, embeddingDim={}, nSeqMax={}",
                    nTokens, embeddingDimension, nSeqMax, e);
            throw new LlamaCppJnaException("Failed to initialize batch", e);
        }
    }

    @Override
    public LlamaBatchNative getOneBatch(int[] tokens) {
        Objects.requireNonNull(tokens);

        logger.debug("Creating one batch with {} tokens", tokens.length);

        Memory tokensMemory = null;

        try {
            // Allocate memory for tokens, int32_t = 4 bytes
            tokensMemory = new Memory(tokens.length * 4L);

            // Copy tokens to memory
            for (int i = 0; i < tokens.length; i++) {
                tokensMemory.setInt(i * 4L, tokens[i]);
            }

            // Create one batch using the library function
            LlamaBatchNative batch = llamaLibrary.llama_batch_get_one(tokensMemory, tokens.length);
            if (batch == null) {
                logger.error("Failed to create batch from tokens");
                throw new LlamaCppJnaException("Failed to create batch from tokens");
            }

            logger.debug("Created one batch from tokens");

            // Return batch - memory ownership is now with native code
            // Do NOT clean up tokensMemory as it will be freed by llama_batch_free
            return batch;
        } catch (Exception e) {
            // Clean up memory only if batch creation failed
            if (tokensMemory != null) {
                tokensMemory.close();
            }

            logger.error("Failed to create one batch from tokens", e);
            throw new LlamaCppJnaException("Failed to create one batch from tokens", e);
        }
    }

    @Override
    public long getBatchSize(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting batch size for context: {}", context.getContextPointer());
            long batchSize = llamaLibrary.llama_n_batch(context.getContextPointer());
            logger.debug("Successfully got batch size: {} for context: {}", batchSize, context.getContextPointer());
            return batchSize;
        } catch (Exception e) {
            logger.error("Error getting batch size for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to get batch size, error: '%s'", e.getMessage()), e);
        }
    }
}
