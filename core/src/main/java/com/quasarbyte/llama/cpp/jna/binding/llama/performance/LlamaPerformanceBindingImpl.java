package com.quasarbyte.llama.cpp.jna.binding.llama.performance;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class LlamaPerformanceBindingImpl implements LlamaPerformanceBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaPerformanceBindingImpl.class);

    private final LlamaLibrary llamaLibrary;

    public LlamaPerformanceBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public LlamaPerformanceContextDataNative getContextPerformance(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        return llamaLibrary.llama_perf_context(context.getContextPointer());
    }

    @Override
    public void printContextPerformance(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        llamaLibrary.llama_perf_context_print(context.getContextPointer());
    }

    @Override
    public void resetContextPerformance(LlamaContext context) {
        llamaLibrary.llama_perf_context_reset(context.getContextPointer());
    }

    @Override
    public LlamaPerformanceSamplerDataNative getSamplerPerformance(LlamaSampler samplerChain) {
        Objects.requireNonNull(samplerChain);
        Objects.requireNonNull(samplerChain.getSamplerPointer());

        if (!ChainRole.HEAD.equals(samplerChain.getChainRole())) {
            logger.error("The sampler parameter is not a head chain sampler");
            throw new LlamaCppJnaException("The sampler parameter is not a head chain sampler");
        }

        return llamaLibrary.llama_perf_sampler(samplerChain.getSamplerPointer());
    }

    @Override
    public void printSamplerPerformance(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());
        llamaLibrary.llama_perf_sampler_print(sampler.getSamplerPointer());
    }

    @Override
    public void resetSamplerPerformance(LlamaSampler samplerChain) {
        Objects.requireNonNull(samplerChain);
        Objects.requireNonNull(samplerChain.getSamplerPointer());

        if (!ChainRole.HEAD.equals(samplerChain.getChainRole())) {
            logger.error("The sampler parameter is not a head chain sampler");
            throw new LlamaCppJnaException("The sampler parameter is not a head chain sampler");
        }

        llamaLibrary.llama_perf_sampler_reset(samplerChain.getSamplerPointer());
    }
}