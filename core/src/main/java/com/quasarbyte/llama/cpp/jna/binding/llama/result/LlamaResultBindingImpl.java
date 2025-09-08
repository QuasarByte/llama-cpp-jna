package com.quasarbyte.llama.cpp.jna.binding.llama.result;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModelNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaVocabularyNative;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.Optional;

/**
 * Implementation of LlamaResultBinding for accessing inference results.
 */
public class LlamaResultBindingImpl implements LlamaResultBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaResultBindingImpl.class);

    private final LlamaLibrary llamaLibrary;

    public LlamaResultBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public Optional<float[]> getLogits(LlamaContext context) {
        try {
            Optional<Pointer> logitsPointer = getLogitsRaw(context);

            if (logitsPointer.isPresent()) {

                int vocabSize = getVocabularySize(context);

                Pointer pointer = logitsPointer.get();

                return Optional.ofNullable(pointer.getFloatArray(0, vocabSize));

            } else {
                return Optional.empty();
            }

        } catch (Exception e) {
            logger.error("Error getting logits", e);
            throw new LlamaCppJnaException(String.format("Error getting logits, error: %s", e.getMessage()), e);
        }
    }

    @Override
    public Optional<float[]> getLogitsAt(LlamaContext context, int tokenIndex) {
        try {
            Optional<Pointer> logitsPointer = getLogitsRawAt(context, tokenIndex);

            if (logitsPointer.isPresent()) {

                int vocabSize = getVocabularySize(context);

                Pointer pointer = logitsPointer.get();

                return Optional.ofNullable(pointer.getFloatArray(0, vocabSize));

            } else {
                return Optional.empty();
            }

        } catch (Exception e) {
            logger.error("Error getting logits at index: {}, error: {}}", tokenIndex, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting logits at index: %d, error: %s", tokenIndex, e.getMessage()), e);
        }
    }

    @Override
    public Optional<float[]> getEmbeddings(LlamaContext context) {
        try {
            Optional<Pointer> embeddingsPointer = getEmbeddingsRaw(context);

            if (embeddingsPointer.isPresent()) {

                Pointer pointer = embeddingsPointer.get();

                int embeddingDim = getEmbeddingDimension(context);
                return Optional.ofNullable(pointer.getFloatArray(0, embeddingDim));

            } else {
                return Optional.empty();
            }

        } catch (Exception e) {
            logger.error("Error getting embeddings, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting embeddings, error: %s", e.getMessage()), e);
        }
    }

    @Override
    public Optional<float[]> getEmbeddingsAt(LlamaContext context, int tokenIndex) {
        try {
            Optional<Pointer> embeddingsPointer = getEmbeddingsRawAt(context, tokenIndex);

            if (embeddingsPointer.isPresent()) {
                int embeddingDim = getEmbeddingDimension(context);

                Pointer pointer = embeddingsPointer.get();

                return Optional.ofNullable( pointer.getFloatArray(0, embeddingDim));

            } else {
                return Optional.empty();
            }

        } catch (Exception e) {
            logger.error("Error getting embeddings at index: {}, error: {}}", tokenIndex, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting embeddings at index: %d, error: %s", tokenIndex, e.getMessage()), e);
        }
    }

    @Override
    public Optional<float[]> getEmbeddingsForSequence(LlamaContext context, int sequenceId) {
        try {
            Optional<Pointer> embeddingsPointer = getEmbeddingsRawForSequence(context, sequenceId);

            if (embeddingsPointer.isPresent()) {

                int embeddingDim = getEmbeddingDimension(context);

                Pointer pointer = embeddingsPointer.get();

                return Optional.ofNullable(pointer.getFloatArray(0, embeddingDim));

            } else {
                return Optional.empty();
            }

        } catch (Exception e) {
            logger.error("Error getting embeddings for sequence: {}, message: {}", sequenceId, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting embeddings for sequence: %d, message: %s", sequenceId, e.getMessage()), e);
        }
    }

    @Override
    public int getVocabularySize(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            // Get model from context and then vocabulary size
            LlamaModelNative modelPointer = llamaLibrary.llama_get_model(context.getContextPointer());

            if (modelPointer == null) {
                logger.error("Error getting vocabulary size, model is null");
                throw new LlamaCppJnaException("Error getting vocabulary size, model is null");
            }

            LlamaVocabularyNative vocabPointer = llamaLibrary.llama_model_get_vocab(modelPointer);

            if (vocabPointer == null) {
                logger.error("Error getting vocabulary size, vocabulary is null");
                throw new LlamaCppJnaException("Error getting vocabulary size, vocabulary is null");
            }

            return llamaLibrary.llama_vocab_n_tokens(vocabPointer);

        } catch (Exception e) {
            logger.error("Error getting vocabulary size, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting vocabulary size, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public int getEmbeddingDimension(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {

            LlamaModelNative modelPointer = llamaLibrary.llama_get_model(context.getContextPointer());

            if (modelPointer == null) {
                logger.error("Error getting embedding dimension, model is null");
                throw new LlamaCppJnaException("Error getting embedding dimension, model is null");
            }

            return llamaLibrary.llama_model_n_embd(modelPointer);

        } catch (Exception e) {
            logger.error("Error getting embedding dimension, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting embedding dimension, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public boolean hasLogits(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        return getLogitsRaw(context).isPresent();
    }

    @Override
    public boolean hasEmbeddings(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        return getEmbeddingsRaw(context).isPresent();
    }

    @Override
    public void setEmbeddingsEnabled(LlamaContext context, boolean enabled) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            llamaLibrary.llama_set_embeddings(context.getContextPointer(), enabled);
            logger.debug("Embeddings {} for context", enabled ? "enabled" : "disabled");
        } catch (Exception e) {
            logger.error("Error setting embeddings enabled: {}, message: {}", enabled, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error setting embeddings enabled: %b, message: %s", enabled, e.getMessage()), e);
        }
    }

    //TODO: return logits.getFloatArray(0, embeddingDim);
    @Override
    public Optional<Pointer> getLogitsRaw(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            return Optional.ofNullable(llamaLibrary.llama_get_logits(context.getContextPointer()));
        } catch (Exception e) {
            logger.error("Error getting raw logits, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting raw logits, message: %s", e.getMessage()), e);
        }
    }

    //TODO: return logits.getFloatArray(0, embeddingDim);
    @Override
    public Optional<Pointer> getLogitsRawAt(LlamaContext context, int tokenIndex) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        if (tokenIndex < 0) {
            logger.error("Error getting raw logits, tokenIndex is negative: {}",  tokenIndex);
            throw new LlamaCppJnaException(String.format( "Error getting raw logits, tokenIndex is negative: %d",  tokenIndex));
        }

        try {
            return Optional.ofNullable(llamaLibrary.llama_get_logits_ith(context.getContextPointer(), tokenIndex));
        } catch (Exception e) {
            logger.error("Error getting raw logits at index: {}, message: {}", tokenIndex, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting raw logits at index: %d, message: %s", tokenIndex, e.getMessage()), e);
        }
    }

    //TODO: implement array conversion
    @Override
    public Optional<Pointer> getEmbeddingsRaw(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            return Optional.ofNullable(llamaLibrary.llama_get_embeddings(context.getContextPointer()));
        } catch (Exception e) {
            logger.error("Error getting raw embeddings, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting raw embeddings, message: %s", e.getMessage()), e);
        }
    }

    //TODO: implement array conversion
    @Override
    public Optional<Pointer> getEmbeddingsRawAt(LlamaContext context, int tokenIndex) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        if (tokenIndex < 0) {
            logger.error("Error getting raw embeddings, tokenIndex is negative: {}",  tokenIndex);
            throw new LlamaCppJnaException(String.format( "Error getting raw embeddings, tokenIndex is negative: %d",  tokenIndex));
        }

        try {
            return Optional.ofNullable(llamaLibrary.llama_get_embeddings_ith(context.getContextPointer(), tokenIndex));
        } catch (Exception e) {
            logger.error("Error getting raw embeddings at index: {}, message: {}", tokenIndex, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting raw embeddings at index: %d, message: %s", tokenIndex, e.getMessage()), e);
        }
    }

    //TODO: implement array conversion
    @Override
    public Optional<Pointer> getEmbeddingsRawForSequence(LlamaContext context, int sequenceId) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            return Optional.ofNullable(llamaLibrary.llama_get_embeddings_seq(context.getContextPointer(), sequenceId));
        } catch (Exception e) {
            logger.error("Error getting raw embeddings for sequence: {}, message: {}", sequenceId, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting raw embeddings for sequence: %d, message: %s", sequenceId, e.getMessage()), e);
        }
    }
}