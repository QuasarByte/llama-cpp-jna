package com.quasarbyte.llama.cpp.jna.binding.llama.optimization;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaOptimizationParams;
import com.sun.jna.Pointer;

/**
 * Service for training and optimization operations on LLaMA models.
 * <p>
 * This binding provides methods for fine-tuning and training LLaMA models
 * using gradient-based optimization techniques.
 * <p>
 * <strong>Note:</strong> Training functionality is experimental and requires
 * specific build configurations of llama.cpp with training support enabled.
 */
public interface LlamaOptimizationBinding {

    /**
     * Initialize optimization context for training.
     * <p>
     * Sets up the optimization environment for training with the specified
     * model and parameters.
     *
     * @param context the context to use for training
     * @param model the model to train
     * @param optimizationParams optimization parameters (learning rate, etc.)
     */
    void initializeOptimization(LlamaContext context, LlamaModel model, LlamaOptimizationParams optimizationParams);

    /**
     * Run one training epoch on a dataset.
     * <p>
     * Performs one complete pass through the training dataset, computing
     * gradients and updating model parameters.
     *
     * @param context the training context
     * @param dataset pointer to the training dataset
     * @param trainResult pointer to store training results
     * @param evalResult pointer to store evaluation results (can be null)
     * @param dataSplitIndex index for data splitting
     * @param trainCallback optional callback for training progress
     * @param evalCallback optional callback for evaluation progress
     */
    void runEpoch(LlamaContext context, Pointer dataset, Pointer trainResult, 
                  Pointer evalResult, long dataSplitIndex, 
                  Pointer trainCallback, Pointer evalCallback);

    /**
     * Check if all model parameters should be included in optimization.
     * <p>
     * This is a filter function that determines which tensors/parameters
     * should be optimized during training.
     *
     * @param tensor pointer to the tensor
     * @param userData user data for the filter function
     * @return true to include tensor in optimization, false to exclude
     */
    boolean shouldOptimizeAllParameters(Pointer tensor, Pointer userData);

    /**
     * Create default optimization parameters.
     * <p>
     * Returns a set of default parameters suitable for fine-tuning
     * LLaMA models.
     *
     * @return default optimization parameters container
     */
    LlamaOptimizationParams createDefaultOptimizationParams();

    /**
     * Set learning rate for optimization.
     *
     * @param optimizationParams the optimization parameters to modify
     * @param learningRate the learning rate (typically 1e-5 to 1e-3)
     */
    void setLearningRate(LlamaOptimizationParams optimizationParams, float learningRate);

    /**
     * Set batch size for optimization.
     *
     * @param optimizationParams the optimization parameters to modify
     * @param batchSize the batch size for training
     */
    void setBatchSize(LlamaOptimizationParams optimizationParams, int batchSize);

    /**
     * Set maximum number of epochs.
     *
     * @param optimizationParams the optimization parameters to modify
     * @param maxEpochs maximum number of training epochs
     */
    void setMaxEpochs(LlamaOptimizationParams optimizationParams, int maxEpochs);

    /**
     * Free optimization parameters.
     *
     * @param optimizationParams the parameters to release/clear
     */
    void freeOptimizationParams(LlamaOptimizationParams optimizationParams);

    /**
     * Check if training/optimization is supported in the current build.
     *
     * @return true if training is supported, false otherwise
     */
    boolean isTrainingSupported();

    /**
     * Get the current optimization/training status.
     *
     * @param context the training context
     * @return status description
     */
    String getOptimizationStatus(LlamaContext context);
}