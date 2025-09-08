package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;

/**
 * Plain Java representation of {@code llama_opt_params}.
 */
public class LlamaOptimizationParams {

    private int contextSize;
    private Pointer parameterFilter;
    private Pointer parameterFilterUserData;
    private Pointer optimizerParameterCallback;
    private Pointer optimizerParameterUserData;
    private int optimizerType;
    private float learningRate = 1e-4f;
    private int batchSize = 1;
    private int maxEpochs = 1;

    public int getContextSize() {
        return contextSize;
    }

    public LlamaOptimizationParams setContextSize(int contextSize) {
        this.contextSize = contextSize;
        return this;
    }

    public Pointer getParameterFilter() {
        return parameterFilter;
    }

    public LlamaOptimizationParams setParameterFilter(Pointer parameterFilter) {
        this.parameterFilter = parameterFilter;
        return this;
    }

    public Pointer getParameterFilterUserData() {
        return parameterFilterUserData;
    }

    public LlamaOptimizationParams setParameterFilterUserData(Pointer parameterFilterUserData) {
        this.parameterFilterUserData = parameterFilterUserData;
        return this;
    }

    public Pointer getOptimizerParameterCallback() {
        return optimizerParameterCallback;
    }

    public LlamaOptimizationParams setOptimizerParameterCallback(Pointer optimizerParameterCallback) {
        this.optimizerParameterCallback = optimizerParameterCallback;
        return this;
    }

    public Pointer getOptimizerParameterUserData() {
        return optimizerParameterUserData;
    }

    public LlamaOptimizationParams setOptimizerParameterUserData(Pointer optimizerParameterUserData) {
        this.optimizerParameterUserData = optimizerParameterUserData;
        return this;
    }

    public int getOptimizerType() {
        return optimizerType;
    }

    public LlamaOptimizationParams setOptimizerType(int optimizerType) {
        this.optimizerType = optimizerType;
        return this;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public LlamaOptimizationParams setLearningRate(float learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public LlamaOptimizationParams setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public int getMaxEpochs() {
        return maxEpochs;
    }

    public LlamaOptimizationParams setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
        return this;
    }

    public LlamaOptParamsNative toNative() {
        LlamaOptParamsNative nativeParams = new LlamaOptParamsNative();
        nativeParams.n_ctx_train = contextSize;
        nativeParams.param_filter = parameterFilter;
        nativeParams.param_filter_ud = parameterFilterUserData;
        nativeParams.get_opt_pars = optimizerParameterCallback;
        nativeParams.get_opt_pars_ud = optimizerParameterUserData;
        nativeParams.optimizer_type = optimizerType;
        nativeParams.write();
        return nativeParams;
    }
}
