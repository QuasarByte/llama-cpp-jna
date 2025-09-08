package com.quasarbyte.llama.cpp.jna.binding.llama.batch;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaBatch;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaBatchNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;

public interface LlamaBatchBinding {
    LlamaBatch createBatch(int[] tokens);

    void freeBatch(LlamaBatch llamaBatch);

    /**
     * Initialize a batch structure for multi-sequence processing.
     *
     * @param nTokens maximum number of tokens
     * @param embeddingDimension embedding dimension (0 if using tokens)
     * @param nSeqMax maximum number of sequences
     * @return initialized batch structure
     */
    LlamaBatchNative initializeBatch(int nTokens, int embeddingDimension, int nSeqMax);

    /**
     * Create a simple batch with tokens for single sequence processing.
     *
     * @param tokens pointer to token array
     * @return initialized batch structure
     */
    LlamaBatchNative getOneBatch(int[] tokens);

    long getBatchSize(LlamaContext context);
}
