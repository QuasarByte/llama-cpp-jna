package com.quasarbyte.llama.cpp.jna.binding.llama.model;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallLongResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.UInt32;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

public class LlamaModelBindingImpl implements LlamaModelBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaModelBindingImpl.class);

    private final LlamaLibrary llamaLibrary;
    private final LlamaStringBufferReader llamaStringBufferReader;

    public LlamaModelBindingImpl(LlamaLibrary llamaLibrary, LlamaStringBufferReader llamaStringBufferReader) {
        this.llamaLibrary = llamaLibrary;
        this.llamaStringBufferReader = llamaStringBufferReader;
    }

    @Override
    public LlamaModel loadFromFile(Path modelPath) {
        return loadFromFile(modelPath, this.llamaLibrary.llama_model_default_params());
    }

    @Override
    public LlamaModel loadFromFile(Path modelPath, LlamaModelParamsNative params) {
        logger.debug("Load model from file");

        Objects.requireNonNull(modelPath);
        Objects.requireNonNull(params);

        Path absoluteModelPath = getExistingAbsoluteFilePath(modelPath);

        String absoluteModelPathAsString = absoluteModelPath.toString();

        LlamaModelNative modelPointer = llamaLibrary.llama_model_load_from_file(absoluteModelPath.toString(), params);

        if (modelPointer == null) {
            logger.error("The Llama model could not be loaded, model path: '{}', model absolute path: '{}'", modelPath, absoluteModelPathAsString);
            throw new LlamaCppJnaException(String.format("Failed to load model from path: '%s', absolute path: '%s'", modelPath, absoluteModelPathAsString));
        }

        logger.debug("Model loaded successfully");

        return new LlamaModel().setModelPointer(modelPointer);
    }

    @Override
    public LlamaModel loadFromFile(Path modelPath, int nGpuLayers) {
        Objects.requireNonNull(modelPath);
        LlamaModelParamsNative params = getDefaultParams();
        params.n_gpu_layers = nGpuLayers;
        return loadFromFile(modelPath, params);
    }

    @Override
    public void freeModel(LlamaModel llamaModel) {
        Objects.requireNonNull(llamaModel);
        Objects.requireNonNull(llamaModel.getModelPointer());
        llamaLibrary.llama_model_free(llamaModel.getModelPointer());
    }

    @Override
    public LlamaModel loadFromSplits(List<Path> paths, LlamaModelParamsNative params) {
        Objects.requireNonNull(paths);
        Objects.requireNonNull(params);

        if (paths.isEmpty()) {
            throw new LlamaCppJnaException("Split paths cannot be empty");
        }

        String[] filesArray = new String[paths.size()];

        for (int i = 0; i < filesArray.length; i++) {
            Path path = paths.get(i);
            Path absolutePath = getExistingAbsoluteFilePath(path);
            String absolutePathAsString = absolutePath.toString();
            filesArray[i] = absolutePathAsString;
        }

        LlamaModelNative modelPointer = llamaLibrary.llama_model_load_from_splits(filesArray, filesArray.length, params);
        
        if (modelPointer == null) {
            logger.error("Failed to load model from splits, result is null");
            throw new LlamaCppJnaException("Failed to load model from splits, result is null");
        }

        LlamaModel llamaModel = new LlamaModel();
        llamaModel.setModelPointer(modelPointer);
        return llamaModel;
    }

    @Override
    public void saveToFile(LlamaModel model, String path) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());
        Objects.requireNonNull(path);

        try {
            llamaLibrary.llama_model_save_to_file(model.getModelPointer(), path);
        } catch (Exception e) {
            logger.error("Cannot save model to '{}', error: {}", path, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Cannot save model to '%s', error: %s", path, e.getMessage()), e);
        }

        logger.debug("Model saved to: {}", path);
    }

    @Override
    public int getRopeType(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_rope_type);
    }

    @Override
    public int getTrainContextSize(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_ctx_train);
    }

    @Override
    public int getEmbeddingDimension(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_embd);
    }

    @Override
    public int getLayerCount(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_layer);
    }

    @Override
    public int getAttentionHeadCount(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_head);
    }

    @Override
    public int getKeyValueHeadCount(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_head_kv);
    }

    @Override
    public int getSlidingWindowAttentionLayers(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_swa);
    }

    @Override
    public float getRopeFrequencyScaleTrain(LlamaModel model) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_rope_freq_scale_train(model.getModelPointer());
    }

    @Override
    public int getClassificationOutputCount(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_n_cls_out);
    }

    @Override
    public String getClassificationLabel(LlamaModel model, UInt32 index) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_cls_label(model.getModelPointer(), index);
    }

    @Override
    public String getMetadata(LlamaModel model, String key, int maxLength) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());

        if (key == null) {
            logger.error("Parameter key cannot be null");
            throw new LlamaCppJnaException("Parameter key cannot be null");
        }

        if (key.trim().isEmpty()) {
            logger.error("Parameter key cannot be blank");
            throw new LlamaCppJnaException("Parameter key cannot be blank");
        }

        if (maxLength < 0) {
            logger.error("Parameter maxLength cannot be negative");
            throw new LlamaCppJnaException("Parameter maxLength cannot be negative");
        }

        byte[] buffer = new byte[maxLength];
        int length = llamaLibrary.llama_model_meta_val_str(model.getModelPointer(), key, buffer, buffer.length);

        if (length < 0) {
            logger.error("Failed to get metadata, length: {}, key: {}, maxLength: {}", length, key, maxLength);
            throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get metadata, length: %d, key: %s, maxLength: %d", length, key, maxLength));
        }

        return llamaStringBufferReader.readString(buffer, length);
    }

    @Override
    public int getMetadataCount(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_meta_count);
    }

    @Override
    public String getMetadataKey(LlamaModel model, int index, int maxLength) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());

        if (index < 0) {
            logger.error("Parameter index cannot be negative");
            throw new LlamaCppJnaException("Parameter index cannot be negative");
        }

        if (maxLength < 0) {
            logger.error("Parameter maxLength cannot be negative");
            throw new LlamaCppJnaException("Parameter maxLength cannot be negative");
        }

        byte[] buffer = new byte[maxLength];
        int length = llamaLibrary.llama_model_meta_key_by_index(model.getModelPointer(), index, buffer, buffer.length);

        if (length < 0) {
            logger.error("Failed to get metadata key by index, length: {}, index: {}, maxLength: {}", length, index, maxLength);
            throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get metadata key by index, length: %d, index: %s, maxLength: %d", length, index, maxLength));
        }

        return llamaStringBufferReader.readString(buffer, length);
    }

    @Override
    public String getMetadataValue(LlamaModel model, int index, int maxLength) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());

        if (index < 0) {
            logger.error("Parameter index cannot be negative");
            throw new LlamaCppJnaException("Parameter index cannot be negative");
        }

        if (maxLength < 0) {
            logger.error("Parameter maxLength cannot be negative");
            throw new LlamaCppJnaException("Parameter maxLength cannot be negative");
        }

        byte[] buffer = new byte[maxLength];
        int length = llamaLibrary.llama_model_meta_val_str_by_index(model.getModelPointer(), index, buffer, buffer.length);

        if (length < 0) {
            logger.error("Failed to get metadata value by index, length: {}, index: {}, maxLength: {}", length, index, maxLength);
            throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get metadata value by index, length: %d, index: %s, maxLength: %d", length, index, maxLength));
        }

        return this.llamaStringBufferReader.readString(buffer, length);
    }

    @Override
    public String getDescription(LlamaModel model, int maxLength) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());

        byte[] buffer = new byte[maxLength];
        int length = llamaLibrary.llama_model_desc(model.getModelPointer(), buffer, buffer.length);

        if (length < 0) {
            logger.error("Failed to get metadata description, length: {}, maxLength: {}", length, maxLength);
            throw new LlamaFunctionCallIntResultException(length, String.format("Failed to get metadata description, length: %d, maxLength: %d", length, maxLength));
        }

        return this.llamaStringBufferReader.readString(buffer,length);
    }

    @Override
    public long getSize(LlamaModel model) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());

        long size = llamaLibrary.llama_model_size(model.getModelPointer());

        if (size < 0) {
            logger.error("Failed to get model size: {}", size);
            throw new LlamaFunctionCallLongResultException(size, String.format("Failed to get model size: %d", size));
        }

        return size;
    }

    @Override
    public long getParameterCount(LlamaModel model) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_n_params(model.getModelPointer());
    }

    @Override
    public boolean hasEncoder(LlamaModel model) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_has_encoder(model.getModelPointer());
    }

    @Override
    public boolean hasDecoder(LlamaModel model) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_has_decoder(model.getModelPointer());
    }

    @Override
    public int getDecoderStartToken(LlamaModel model) {
        return validateAndGetModelInfo(model, llamaLibrary::llama_model_decoder_start_token);
    }

    @Override
    public boolean isRecurrent(LlamaModel model) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_is_recurrent(model.getModelPointer());
    }

    @Override
    public boolean isDiffusion(LlamaModel model) {
        Objects.requireNonNull(model, "model is null");
        
        if (model.getModelPointer() == null) {
            throw new LlamaCppJnaException("Model is invalid");
        }

        return llamaLibrary.llama_model_is_diffusion(model.getModelPointer());
    }

    @Override
    public void quantizeModel(Path inputPath, Path outputPath, LlamaModelQuantizeParamsNative quantizationParams) {
        Objects.requireNonNull(inputPath);
        Objects.requireNonNull(outputPath);
        Objects.requireNonNull(quantizationParams);

        String inputPathAsString = getExistingAbsoluteFilePath(inputPath).toString();
        String outputPathAsString = outputPath.toAbsolutePath().toString();

        UInt32 result = llamaLibrary.llama_model_quantize(inputPathAsString, outputPathAsString, quantizationParams);

        if (result.longValue() != 0) {
            logger.error("Failed to quantize model, error code is {}", result);
            throw new LlamaFunctionCallLongResultException(result.longValue(), String.format("Failed to quantize model, error code is %d", result.longValue()));
        }
    }

    @Override
    public LlamaModelParamsNative getDefaultParams() {
        final LlamaModelParamsNative params;

        try {
            params = llamaLibrary.llama_model_default_params();
        } catch (Exception e) {
            logger.error("The Llama model default parameters could not be loaded, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("The Llama model default parameters could not be loaded, error: %s", e.getMessage()), e);
        }

        if (params == null) {
            logger.error("The Llama model default parameters could not be loaded, result is null");
            throw new LlamaCppJnaException("The Llama model default parameters could not be loaded, result is null");
        }

        return params;
    }

    /**
     * Helper method to validate model and get integer information.
     */
    private int validateAndGetModelInfo(LlamaModel model, ModelInfoFunction function) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());
        Objects.requireNonNull(function);
        return function.apply(model.getModelPointer());
    }

    /**
     * Functional interface for model information functions that return int.
     */
    @FunctionalInterface
    private interface ModelInfoFunction {
        int apply(LlamaModelNative modelPointer);
    }

    private static Path getExistingAbsoluteFilePath(Path path) {
        Objects.requireNonNull(path);

        Path absolutePath = path.toAbsolutePath();

        if (Files.exists(absolutePath)) {

            if (Files.isDirectory(absolutePath)) {
                logger.error("Path should be a file not a directory");
                throw new LlamaCppJnaException("Path should be a file not a directory");
            }

            if (!Files.isRegularFile(absolutePath)) {
                logger.error("Path should be a file");
                throw new LlamaCppJnaException("Path should be a file");
            }

        } else {
            logger.error("File not found");
            throw new LlamaCppJnaException("File not found");
        }
        return absolutePath;
    }
}