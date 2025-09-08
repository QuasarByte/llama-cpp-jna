package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of model splitting modes for multi-GPU setups supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_split_mode} enumeration from llama.h.
 * These modes determine how model layers and data are distributed across multiple GPUs.
 */
public enum LlamaSplitMode {
    
    /** No splitting - use single GPU */
    LLAMA_SPLIT_MODE_NONE(0),
    
    /** Layer splitting - split layers and KV cache across GPUs */
    LLAMA_SPLIT_MODE_LAYER(1),
    
    /** Row splitting - split layers and KV cache across GPUs, use tensor parallelism if supported */
    LLAMA_SPLIT_MODE_ROW(2);
    
    private final int value;
    
    LlamaSplitMode(int value) {
        this.value = value;
    }
    
    /**
     * Gets the integer value corresponding to the native enum value.
     * 
     * @return the integer value
     */
    public int getValue() {
        return value;
    }
    
    /**
     * Returns the enum constant for the given integer value.
     * 
     * @param value the integer value from the native enum
     * @return the corresponding enum constant
     * @throws IllegalArgumentException if no enum constant matches the value
     */
    public static LlamaSplitMode fromValue(int value) {
        for (LlamaSplitMode mode : values()) {
            if (mode.value == value) {
                return mode;
            }
        }
        throw new IllegalArgumentException("Unknown split mode value: " + value);
    }
    
    /**
     * Checks if this is single GPU mode (no splitting).
     * 
     * @return true if using single GPU
     */
    public boolean isSingleGpu() {
        return this == LLAMA_SPLIT_MODE_NONE;
    }
    
    /**
     * Checks if this mode uses multiple GPUs.
     * 
     * @return true if using multiple GPUs
     */
    public boolean isMultiGpu() {
        return this != LLAMA_SPLIT_MODE_NONE;
    }
    
    /**
     * Checks if this is layer-based splitting.
     * In layer splitting, entire layers are assigned to different GPUs.
     * 
     * @return true if using layer-based splitting
     */
    public boolean isLayerSplitting() {
        return this == LLAMA_SPLIT_MODE_LAYER;
    }
    
    /**
     * Checks if this is row-based splitting with tensor parallelism.
     * In row splitting, individual layers are split across GPUs at the tensor level.
     * 
     * @return true if using row-based splitting with tensor parallelism
     */
    public boolean isRowSplitting() {
        return this == LLAMA_SPLIT_MODE_ROW;
    }
    
    /**
     * Checks if this mode supports tensor parallelism.
     * Tensor parallelism allows splitting individual operations across GPUs.
     * 
     * @return true if tensor parallelism is supported
     */
    public boolean supportsTensorParallelism() {
        return this == LLAMA_SPLIT_MODE_ROW;
    }
    
    /**
     * Gets the recommended minimum number of GPUs for this split mode.
     * 
     * @return minimum recommended GPU count
     */
    public int getMinimumGpuCount() {
        switch (this) {
            case LLAMA_SPLIT_MODE_NONE:
                return 1;
            case LLAMA_SPLIT_MODE_LAYER:
            case LLAMA_SPLIT_MODE_ROW:
                return 2;
            default:
                return 1;
        }
    }
}