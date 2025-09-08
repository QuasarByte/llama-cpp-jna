package com.quasarbyte.llama.cpp.jna.binding.llama.adapter;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.Optional;

import static com.quasarbyte.llama.cpp.jna.model.library.LlamaAdapterServiceConstants.*;

/**
 * Implementation of LlamaAdapterBinding for managing LoRA adapters and control vectors.
 */
public class LlamaAdapterBindingImpl implements LlamaAdapterBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaAdapterBindingImpl.class);

    private final LlamaLibrary llamaLibrary;
    private final LlamaStringBufferReader llamaStringBufferReader;
    private final LlamaAdapterServiceSettings llamaAdapterServiceSettings;

    public LlamaAdapterBindingImpl(LlamaLibrary llamaLibrary,
                                   LlamaStringBufferReader llamaStringBufferReader,
                                   LlamaAdapterServiceSettings llamaAdapterServiceSettings) {

        this.llamaLibrary = llamaLibrary;
        this.llamaStringBufferReader = llamaStringBufferReader;
        this.llamaAdapterServiceSettings = llamaAdapterServiceSettings;
    }

    @Override
    public LlamaAdapter loadLoraAdapter(LlamaModel model, String adapterPath) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());
        Objects.requireNonNull(adapterPath);

        try {
            logger.debug("Loading LoRA adapter from path: {}", adapterPath);
            LlamaAdapterNative adapterPointer = llamaLibrary.llama_adapter_lora_init(model.getModelPointer(), adapterPath);
            
            if (adapterPointer == null) {
                logger.error("Failed to load LoRA adapter from: {}", adapterPath);
                throw new LlamaCppJnaException(String.format("Failed to load LoRA adapter from: %s", adapterPath));
            }

            logger.debug("Successfully loaded LoRA adapter from: {}", adapterPath);
            return new LlamaAdapter(adapterPointer);
            
        } catch (Exception e) {
            logger.error("Error loading LoRA adapter from path: {}", adapterPath, e);
            throw new LlamaCppJnaException(String.format("Error loading LoRA adapter from: %s", adapterPath), e);
        }
    }

    @Override
    public String getAdapterMetadata(LlamaAdapter adapter, String key) {
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());
        Objects.requireNonNull(key);

        if (key.trim().isEmpty()) {
            logger.error("Key can not be blank");
            throw new LlamaCppJnaException("Key can not be blank");
        }

        try {
            Integer bufferSize = Optional.ofNullable( llamaAdapterServiceSettings.getAdapterMetadataBufferSize())
                    .orElse(GET_ADAPTER_METADATA_BUFFER_SIZE);

            byte[] buffer = new byte[bufferSize]; // Reasonable buffer size for metadata
            int length = llamaLibrary.llama_adapter_meta_val_str(adapter.getNativePointer(), key, buffer, buffer.length);
            
            if (length < 0) {
                logger.error("Failed to get adapter metadata value as string by key {}, length: {}", key, length);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get adapter metadata value as string by key '%s', length: %d", key, length));
            }

            return llamaStringBufferReader.readString(buffer, length);
        } catch (Exception e) {
            logger.error("Error getting adapter metadata for key: {}", key, e);
            throw new LlamaCppJnaException(String.format("Error getting adapter metadata for key '%s', message: %s", key, e.getMessage()), e);
        }
    }

    @Override
    public int getAdapterMetadataCount(LlamaAdapter adapter) {
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        try {
            return llamaLibrary.llama_adapter_meta_count(adapter.getNativePointer());
        } catch (Exception e) {
            logger.error("Error getting adapter metadata count, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting adapter metadata count, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public String getAdapterMetadataKey(LlamaAdapter adapter, int index) {
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        if (index < 0) {
            logger.error("Index is negative");
            throw new LlamaCppJnaException("Index is negative");
        }

        try {
            Integer bufferSize = Optional.ofNullable(llamaAdapterServiceSettings.getAdapterMetadataKeyBufferSize())
                    .orElse(GET_ADAPTER_METADATA_KEY_BUFFER_SIZE);

            byte[] buffer = new byte[bufferSize];
            int length = llamaLibrary.llama_adapter_meta_key_by_index(adapter.getNativePointer(), index, buffer, buffer.length);
            
            if (length < 0) {
                logger.error("Failed to get adapter metadata key by index, result: {}", length);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get adapter metadata key by index, result: %d", length));
            }

            return llamaStringBufferReader.readString(buffer, length);

        } catch (Exception e) {
            logger.error("Error getting adapter metadata key at index: {}, message: {}", index, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting adapter metadata key at index: %d, message: %s", index, e.getMessage()), e);
        }
    }

    @Override
    public String getAdapterMetadataValue(LlamaAdapter adapter, int index) {
        try {
            if (adapter == null || adapter.getNativePointer() == null || index < 0) {
                return null;
            }

            Integer bufferSize = Optional.ofNullable(llamaAdapterServiceSettings.getAdapterMetadataValueBufferSize())
                    .orElse(GET_ADAPTER_METADATA_VALUE_BUFFER_SIZE);

            byte[] buffer = new byte[bufferSize];
            int length = llamaLibrary.llama_adapter_meta_val_str_by_index(adapter.getNativePointer(), index, buffer, buffer.length);

            if (length < 0) {
                logger.error("Failed to get adapter metadata value at index: {}, result: {}", index, length);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get adapter metadata value at index: %d, result: %d", index, length));
            }

            return llamaStringBufferReader.readString(buffer, length);
            
        } catch (Exception e) {
            logger.error("Failed to get adapter metadata value at index: {}, message: {}", index, e.getMessage());
            throw new LlamaCppJnaException(String.format("Failed to get adapter metadata value at index: %d, message: %s", index, e.getMessage()), e);
        }
    }

    @Override
    public long getAloraInvocationTokenCount(LlamaAdapter adapter) {
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        try {
            return llamaLibrary.llama_adapter_get_alora_n_invocation_tokens(adapter.getNativePointer());
        } catch (Exception e) {
            logger.error("Error getting ALoRA invocation token count, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting ALoRA invocation token count, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public int[] getAloraInvocationTokens(LlamaAdapter adapter) {
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        try {

            long tokenCount = getAloraInvocationTokenCount(adapter);

            Pointer tokensPointer = llamaLibrary.llama_adapter_get_alora_invocation_tokens(adapter.getNativePointer());

            if (tokensPointer == null) {
                logger.error("Error getting ALoRA invocation tokens");
                throw new LlamaCppJnaException("Error getting ALoRA invocation tokens");
            }

            return tokensPointer.getIntArray(0, (int) tokenCount);
            
        } catch (Exception e) {
            logger.error("Error getting ALoRA invocation tokens, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error getting ALoRA invocation tokens, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public void setLoraAdapter(LlamaContext context, LlamaAdapter adapter, float scale) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        try {

            int result = llamaLibrary.llama_set_adapter_lora(context.getContextPointer(), adapter.getNativePointer(), scale);

            if (result != 0) {
                logger.error("Error setting LoRA adapter with scale: {}, result: {}", scale, result);
                throw new LlamaFunctionCallIntResultException(result, String.format("Error setting LoRA adapter with scale: %f, result: %d", scale, result));
            }

        } catch (Exception e) {
            logger.error("Error setting LoRA adapter with scale: {}, message: {}", scale, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error setting LoRA adapter with scale: %f, message: %s", scale, e.getMessage()), e);
        }
    }

    @Override
    public void removeLoraAdapter(LlamaContext context, LlamaAdapter adapter) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        try {

            int result = llamaLibrary.llama_rm_adapter_lora(context.getContextPointer(), adapter.getNativePointer());

            if (result != 0) {
                logger.error("Error removing LoRA adapter, result: {}", result);
                throw new LlamaFunctionCallIntResultException(result, String.format("Error removing LoRA adapter, result: %d", result));
            }

        } catch (Exception e) {
            logger.error("Error removing LoRA adapter, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error removing LoRA adapter, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public void clearLoraAdapters(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        try {
            llamaLibrary.llama_clear_adapter_lora(context.getContextPointer());
        } catch (Exception e) {
            logger.error("Failed to clear LoRA adapters, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to clear LoRA adapters, error: %s", e.getMessage()), e);
        }
    }

    @Override
    public void applyControlVector(LlamaContext context, Pointer data, long length,
                                    int embeddingSize, int startLayer, int endLayer) {

        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        if (length < 0) {
            logger.error("Error applying control vector, length less than zero: {}", length);
            throw new LlamaCppJnaException(String.format("Error applying control vector, length less than zero: %d", length));
        }

        if (embeddingSize < 0) {
            logger.error("Error applying control vector, embeddingSize less than zero: {}", embeddingSize);
            throw new LlamaCppJnaException(String.format("Error applying control vector, embeddingSize less than zero: %d", embeddingSize));
        }

        if (startLayer < 0) {
            logger.error("Error applying control vector, startLayer less than zero: {}", startLayer);
            throw new LlamaCppJnaException(String.format("Error applying control vector, startLayer less than zero: %d", startLayer));
        }

        if (endLayer < 0) {
            logger.error("Error applying control vector, endLayer less than zero: {}", endLayer);
            throw new LlamaCppJnaException(String.format("Error applying control vector, endLayer less than zero: %d", endLayer));
        }

        try {

            int result = llamaLibrary.llama_apply_adapter_cvec(context.getContextPointer(), data, length, embeddingSize, startLayer, endLayer);

            if (result != 0) {
                logger.error("Error applying control vector, result: {}", result);
                throw new LlamaFunctionCallIntResultException(result, String.format("Error applying control vector, result: %d", result));
            }

        } catch (Exception e) {
            logger.error("Error applying control vector, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error applying control vector, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public void clearControlVector(LlamaContext context) {
        applyControlVector(context, null, 0, 0, 0, 0);
    }

    @Override
    public void freeAdapter(LlamaAdapter adapter) {
        Objects.requireNonNull(adapter);
        Objects.requireNonNull(adapter.getNativePointer());

        try {
            llamaLibrary.llama_adapter_lora_free(adapter.getNativePointer());
            logger.debug("Freed LoRA adapter");
        } catch (Exception e) {
            logger.error("Failed to free LoRA adapter, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to free LoRA adapter, message: %s", e.getMessage()), e);
        }
    }
}