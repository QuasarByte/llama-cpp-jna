package com.quasarbyte.llama.cpp.jna.binding.llama.model;

import com.quasarbyte.llama.cpp.jna.library.declaration.UInt32;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModelParamsNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModelQuantizeParamsNative;

import java.nio.file.Path;
import java.util.List;

public interface LlamaModelBinding {

    // Model Loading Functions
    LlamaModel loadFromFile(Path modelPath);

    LlamaModel loadFromFile(Path modelPath, LlamaModelParamsNative params);

    LlamaModel loadFromFile(Path modelPath, int nGpuLayers);

    LlamaModel loadFromSplits(List<Path> paths, LlamaModelParamsNative params);

    void saveToFile(LlamaModel model, String path);

    void freeModel(LlamaModel llamaModel);

    // Model Information Functions
    int getRopeType(LlamaModel model);

    int getTrainContextSize(LlamaModel model);

    int getEmbeddingDimension(LlamaModel model);

    int getLayerCount(LlamaModel model);

    int getAttentionHeadCount(LlamaModel model);

    int getKeyValueHeadCount(LlamaModel model);

    int getSlidingWindowAttentionLayers(LlamaModel model);

    float getRopeFrequencyScaleTrain(LlamaModel model);

    int getClassificationOutputCount(LlamaModel model);

    String getClassificationLabel(LlamaModel model, UInt32 index);

    String getMetadata(LlamaModel model, String key, int maxLength);

    int getMetadataCount(LlamaModel model);

    String getMetadataKey(LlamaModel model, int index, int maxLength);

    String getMetadataValue(LlamaModel model, int index, int maxLength);

    String getDescription(LlamaModel model, int maxLength);

    long getSize(LlamaModel model);

    long getParameterCount(LlamaModel model);

    boolean hasEncoder(LlamaModel model);

    boolean hasDecoder(LlamaModel model);

    int getDecoderStartToken(LlamaModel model);

    boolean isRecurrent(LlamaModel model);

    boolean isDiffusion(LlamaModel model);

    // Model Quantization
    void quantizeModel(Path inputPath, Path outputPath, LlamaModelQuantizeParamsNative quantizationParams);

    // Default Parameters
    LlamaModelParamsNative getDefaultParams();
}
