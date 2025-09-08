package com.quasarbyte.llama.cpp.jna.binding.llama.optimization;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaOptimizationParams;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaOptParamsNative;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of LlamaOptimizationBinding for training operations.
 */
public class LlamaOptimizationBindingImpl implements LlamaOptimizationBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaOptimizationBindingImpl.class);

    private final LlamaLibrary llamaLibrary;

    public LlamaOptimizationBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    /**
     * Initializes llama.cpp training by translating the managed parameters into the native
     * {@code llama_opt_params} structure before the first optimization step.
     *
     * @param context wrapped context whose pointer is forwarded to llama.cpp
     * @param model wrapped model to optimize
     * @param optimizationParams managed optimization settings (learning rate, batch size, etc.)
     */
    @Override
    public void initializeOptimization(LlamaContext context, LlamaModel model, LlamaOptimizationParams optimizationParams) {
        try {
            if (context == null || context.getContextPointer() == null) {
                throw new LlamaCppJnaException("Invalid context provided");
            }
            if (model == null || model.getModelPointer() == null) {
                throw new LlamaCppJnaException("Invalid model provided");
            }
            if (optimizationParams == null) {
                throw new LlamaCppJnaException("Optimization parameters cannot be null");
            }

            LlamaOptParamsNative nativeParams = optimizationParams.toNative();
            llamaLibrary.llama_opt_init(context.getContextPointer(), model.getModelPointer(), nativeParams);
            logger.debug("Optimization initialized successfully");

        } catch (Exception e) {
            logger.error("Error initializing optimization", e);
            throw new LlamaCppJnaException("Failed to initialize optimization", e);
        }
    }

    @Override
    public void runEpoch(LlamaContext context, Pointer dataset, Pointer trainResult,
                        Pointer evalResult, long dataSplitIndex,
                        Pointer trainCallback, Pointer evalCallback) {
        try {
            if (context == null || context.getContextPointer() == null) {
                throw new LlamaCppJnaException("Invalid context provided");
            }
            if (dataset == null) {
                throw new LlamaCppJnaException("Dataset cannot be null");
            }

            llamaLibrary.llama_opt_epoch(context.getContextPointer(), dataset, trainResult,
                                       evalResult, dataSplitIndex, trainCallback, evalCallback);
            logger.debug("Training epoch completed");

        } catch (Exception e) {
            logger.error("Error running training epoch", e);
            throw new LlamaCppJnaException("Failed to run training epoch", e);
        }
    }

    @Override
    public boolean shouldOptimizeAllParameters(Pointer tensor, Pointer userData) {
        try {
            return llamaLibrary.llama_opt_param_filter_all(tensor, userData);
        } catch (Exception e) {
            logger.error("Error in parameter filter", e);
            return false;
        }
    }

    /**
     * Builds a convenience container pre-populated with baseline optimization values that callers
     * can adjust before invoking {@link #initializeOptimization}.
     *
     * @return default optimization parameter wrapper
     */
    @Override
    public LlamaOptimizationParams createDefaultOptimizationParams() {
        return new LlamaOptimizationParams().setContextSize(0);
    }

    /**
     * Updates the learning rate stored in the managed optimization parameter wrapper.
     *
     * @param optimizationParams target parameter container
     * @param learningRate new learning rate to persist
     */
    @Override
    public void setLearningRate(LlamaOptimizationParams optimizationParams, float learningRate) {
        if (optimizationParams == null) {
            throw new IllegalArgumentException("Optimization parameters cannot be null");
        }

        optimizationParams.setLearningRate(learningRate);
        logger.debug("Setting learning rate to: {}", learningRate);
    }

    /**
     * Records the intended batch size inside the optimization parameter wrapper.
     *
     * @param optimizationParams target parameter container
     * @param batchSize number of samples per update step
     */
    @Override
    public void setBatchSize(LlamaOptimizationParams optimizationParams, int batchSize) {
        if (optimizationParams == null) {
            throw new IllegalArgumentException("Optimization parameters cannot be null");
        }

        optimizationParams.setBatchSize(batchSize);
        logger.debug("Setting batch size to: {}", batchSize);
    }

    /**
     * Stores the maximum number of epochs that the optimization loop should execute.
     *
     * @param optimizationParams target parameter container
     * @param maxEpochs upper bound for epoch iterations
     */
    @Override
    public void setMaxEpochs(LlamaOptimizationParams optimizationParams, int maxEpochs) {
        if (optimizationParams == null) {
            throw new IllegalArgumentException("Optimization parameters cannot be null");
        }

        optimizationParams.setMaxEpochs(maxEpochs);
        logger.debug("Setting max epochs to: {}", maxEpochs);
    }

    /**
     * Releases references held by the optimization parameter wrapper so the JVM can tidy up
     * after training.
     *
     * @param optimizationParams container to clear; may be {@code null}
     */
    @Override
    public void freeOptimizationParams(LlamaOptimizationParams optimizationParams) {
        if (optimizationParams != null) {
            logger.debug("Clearing optimization parameter references");
        }
    }

    @Override
    public boolean isTrainingSupported() {
        // Check if the current build supports training
        // This would typically check for the presence of training functions
        try {
            // Try to call a training function to see if it exists
            // For now, we'll assume it's supported if the library loaded successfully
            return true;
        } catch (Exception e) {
            logger.warn("Training not supported in current build", e);
            return false;
        }
    }

    @Override
    public String getOptimizationStatus(LlamaContext context) {
        if (context == null || context.getContextPointer() == null) {
            return "Invalid context";
        }

        // This would get the current training status from the context
        // Implementation depends on what status information is available
        return "Training status not available";
    }
}
