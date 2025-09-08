package com.quasarbyte.llama.cpp.jna.binding.llama.performance;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaPerformanceContextDataNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaPerformanceSamplerDataNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaSampler;

public interface LlamaPerformanceBinding {

    LlamaPerformanceContextDataNative getContextPerformance(LlamaContext context);

    void printContextPerformance(LlamaContext context);

    void resetContextPerformance(LlamaContext context);

    LlamaPerformanceSamplerDataNative getSamplerPerformance(LlamaSampler samplerChain);

    void printSamplerPerformance(LlamaSampler sampler);

    void resetSamplerPerformance(LlamaSampler samplerChain);
}