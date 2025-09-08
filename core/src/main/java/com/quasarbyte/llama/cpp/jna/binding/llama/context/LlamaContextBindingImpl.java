package com.quasarbyte.llama.cpp.jna.binding.llama.context;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContextNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContextParamsNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class LlamaContextBindingImpl implements LlamaContextBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaContextBindingImpl.class);
    private final LlamaLibrary llamaLibrary;

    public LlamaContextBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public LlamaContext create(LlamaModel model) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());
        logger.debug("Creating context with default parameters for model: {}", model.getModelPointer());
        return create(model, llamaLibrary.llama_context_default_params());
    }

    @Override
    public LlamaContext create(LlamaModel model, LlamaContextParamsNative params) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());
        Objects.requireNonNull(params);

        try {
            logger.debug("Creating context from model: {}, n_ctx={}, n_batch={}", 
                        model.getModelPointer(), params.n_ctx, params.n_batch);

            LlamaContextNative contextPointer = llamaLibrary.llama_init_from_model(model.getModelPointer(), params);
            if (contextPointer == null) {
                logger.error("Failed to create context from model: {}", model.getModelPointer());
                throw new LlamaCppJnaException("Failed to create context from model");
            }
            
            logger.info("Successfully created context: {}", contextPointer);
            return new LlamaContext(contextPointer);
        } catch (Exception e) {
            logger.error("Error creating context from model: {}", model.getModelPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to create context from model, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public LlamaContext create(LlamaModel model, int nCtx, int nBatch, boolean noPerf) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());

        logger.debug("Creating context with custom parameters: nCtx={}, nBatch={}, noPerf={}", nCtx, nBatch, noPerf);

        final LlamaContextParamsNative params;

        try {
            logger.debug("Getting default context parameters");
            params = llamaLibrary.llama_context_default_params();
        } catch (Exception e) {
            logger.error("Failed to get default context parameters: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to get default context parameters, error: '%s'", e.getMessage()), e);
        }

        params.n_ctx = nCtx;
        params.n_batch = nBatch;
        params.no_perf = (byte) (noPerf ? 1 : 0);

        return create(model, params);
    }

    @Override
    public long getContextSize(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting context size for context: {}", context.getContextPointer());
            long contextSize = llamaLibrary.llama_n_ctx(context.getContextPointer());
            logger.debug("Successfully got context size: {} for context: {}", contextSize, context.getContextPointer());
            return contextSize;
        } catch (Exception e) {
            logger.error("Error getting context size for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Error getting context size, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public long getMicroBatchSize(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting micro-batch size for context: {}", context.getContextPointer());
            long microBatchSize = llamaLibrary.llama_n_ubatch(context.getContextPointer());
            logger.debug("Successfully got micro-batch size: {} for context: {}", microBatchSize, context.getContextPointer());
            return microBatchSize;
        } catch (Exception e) {
            logger.error("Error getting micro-batch size for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Error getting micro-batch size, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public long getMaxNumberOfSequences(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting max number of sequences for context: {}", context.getContextPointer());
            long maxSequences = llamaLibrary.llama_n_seq_max(context.getContextPointer());
            logger.debug("Successfully got max number of sequences: {} for context: {}", maxSequences, context.getContextPointer());
            return maxSequences;
        } catch (Exception e) {
            logger.error("Error getting max sequences for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Error getting max sequences, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void printPerformance(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Printing performance for context: {}", context.getContextPointer());
            llamaLibrary.llama_perf_context_print(context.getContextPointer());
        } catch (Exception e) {
            logger.error("Error printing performance for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to print performance, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void freeContext(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Freeing context: {}", context.getContextPointer());
            llamaLibrary.llama_free(context.getContextPointer());
            logger.info("Successfully freed context: {}", context.getContextPointer());
        } catch (Exception e) {
            logger.error("Error freeing context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to free context, error: '%s'", e.getMessage()), e);
        }
    }

    // Context Configuration Functions

    @Override
    public int getPoolingType(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting pooling type for context: {}", context.getContextPointer());
            int poolingType = llamaLibrary.llama_pooling_type(context.getContextPointer());
            logger.debug("Successfully got pooling type: {} for context: {}", poolingType, context.getContextPointer());
            return poolingType;
        } catch (Exception e) {
            logger.error("Error getting pooling type for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Error getting pooling type, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void setThreadCount(LlamaContext context, int nThreads, int nThreadsBatch) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        if (nThreads < 1) {
            throw new LlamaCppJnaException("Thread count must be positive");
        }

        if (nThreadsBatch < 1) {
            throw new LlamaCppJnaException("Threads batch count must be positive");
        }

        try {
            logger.debug("Setting thread count for context: {}, nThreads={}, nThreadsBatch={}", context.getContextPointer(), nThreads, nThreadsBatch);
            llamaLibrary.llama_set_n_threads(context.getContextPointer(), nThreads, nThreadsBatch);
            logger.debug("Successfully set thread count for context: {}, nThreads={}, nThreadsBatch={}", context.getContextPointer(), nThreads, nThreadsBatch);
        } catch (Exception e) {
            logger.error("Error setting thread count for context: {}, nThreads={}, nThreadsBatch={}", context.getContextPointer(), nThreads, nThreadsBatch, e);
            throw new LlamaCppJnaException(String.format("Failed to set thread count, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public int getThreadCount(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting thread count for context: {}", context.getContextPointer());
            int threadCount = llamaLibrary.llama_n_threads(context.getContextPointer());
            logger.debug("Successfully got thread count: {} for context: {}", threadCount, context.getContextPointer());
            return threadCount;
        } catch (Exception e) {
            logger.error("Error getting thread count for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to get thread count, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public int getBatchThreadCount(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Getting batch thread count for context: {}", context.getContextPointer());
            int batchThreadCount = llamaLibrary.llama_n_threads_batch(context.getContextPointer());
            logger.debug("Successfully got batch thread count: {} for context: {}", batchThreadCount, context.getContextPointer());
            return batchThreadCount;
        } catch (Exception e) {
            logger.error("Error getting batch thread count for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to get batch processing threads count, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void setEmbeddings(LlamaContext context, boolean embeddings) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Setting embeddings for context: {}, embeddings={}", context.getContextPointer(), embeddings);
            llamaLibrary.llama_set_embeddings(context.getContextPointer(), embeddings);
            logger.debug("Successfully set embeddings for context: {}, embeddings={}", context.getContextPointer(), embeddings);
        } catch (Exception e) {
            logger.error("Error setting embeddings for context: {}, embeddings={}", context.getContextPointer(), embeddings, e);
            throw new LlamaCppJnaException(String.format("Failed to set embeddings, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void setCausalAttention(LlamaContext context, boolean causalAttention) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Setting causal attention for context: {}, causalAttention={}", context.getContextPointer(), causalAttention);
            llamaLibrary.llama_set_causal_attn(context.getContextPointer(), causalAttention);
            logger.debug("Successfully set causal attention for context: {}, causalAttention={}", context.getContextPointer(), causalAttention);
        } catch (Exception e) {
            logger.error("Error setting causal attention for context: {}, causalAttention={}", context.getContextPointer(), causalAttention, e);
            throw new LlamaCppJnaException(String.format("Failed to set causal attention, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void setWarmup(LlamaContext context, boolean warmup) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Setting warmup for context: {}, warmup={}", context.getContextPointer(), warmup);
            llamaLibrary.llama_set_warmup(context.getContextPointer(), warmup);
            logger.debug("Successfully set warmup for context: {}, warmup={}", context.getContextPointer(), warmup);
        } catch (Exception e) {
            logger.error("Error setting warmup for context: {}, warmup={}", context.getContextPointer(), warmup, e);
            throw new LlamaCppJnaException(String.format("Failed to set warmup, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void setAbortCallback(LlamaContext context, Pointer abortCallback, Pointer abortCallbackData) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Setting abort callback for context: {}", context.getContextPointer());
            llamaLibrary.llama_set_abort_callback(context.getContextPointer(), abortCallback, abortCallbackData);
            logger.debug("Successfully set abort callback for context: {}", context.getContextPointer());
        } catch (Exception e) {
            logger.error("Error setting abort callback for context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Error setting abort callback, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public void synchronize(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            logger.debug("Synchronizing context: {}", context.getContextPointer());
            llamaLibrary.llama_synchronize(context.getContextPointer());
            logger.debug("Successfully synchronized context: {}", context.getContextPointer());
        } catch (Exception e) {
            logger.error("Error synchronizing context: {}", context.getContextPointer(), e);
            throw new LlamaCppJnaException(String.format("Failed to synchronize context, error: '%s'", e.getMessage()), e);
        }
    }

}
