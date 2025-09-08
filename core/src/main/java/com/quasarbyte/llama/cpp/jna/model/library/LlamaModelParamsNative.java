package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import com.quasarbyte.llama.cpp.jna.model.library.ggml.GgmlBackendBufferTypeNative;

/**
 * Native structure for llama model parameters.
 * <p>
 * This corresponds to the {@code llama_model_params} structure from llama.h.
 * Used for configuring model loading with {@code llama_model_load_from_file()}.
 * <p>
 * <strong>GPU Offloading:</strong> Controls which layers are offloaded to GPU memory
 * and how the model is distributed across multiple GPUs.
 * <p>
 * <strong>Memory Management:</strong> Provides fine-grained control over memory mapping,
 * locking, and buffer allocation strategies.
 * <p>
 * <strong>Loading Behavior:</strong> Supports vocabulary-only loading, tensor validation,
 * and progress callback monitoring during model loading operations.
 */
public class LlamaModelParamsNative extends Structure implements Structure.ByValue {
    /**
     * NULL-terminated list of devices to use for GPU offloading.
     * <p>
     * If set to NULL, all available devices will be used automatically.
     * Otherwise, specifies which specific GPU devices should be used for
     * model layer offloading. Type: {@code ggml_backend_dev_t*}
     */
    public Pointer devices = null;

    /**
     * NULL-terminated list of buffer type overrides for specific tensor patterns.
     * <p>
     * Allows fine-grained control over which buffer types are used for tensors
     * matching specific patterns. Used for advanced memory optimization scenarios.
     * Type: {@code const struct llama_model_tensor_buft_override*}
     */
    public Pointer tensor_buft_overrides = null;

    /**
     * Number of model layers to store in GPU VRAM.
     * <p>
     * Controls how many transformer layers are offloaded to GPU memory.
     * Setting to 0 disables GPU offloading entirely. Higher values increase
     * GPU memory usage but can significantly improve inference speed.
     * <p>
     * <strong>Typical Values:</strong>
     * <ul>
     *   <li>0: CPU-only inference</li>
     *   <li>-1: Offload all layers to GPU</li>
     *   <li>1-N: Offload specific number of layers</li>
     * </ul>
     */
    public int n_gpu_layers = 0;
    /**
     * Strategy for splitting the model across multiple GPUs.
     * <p>
     * Determines how model layers are distributed when multiple GPUs are available.
     * Uses values from {@code enum llama_split_mode}:
     * <ul>
     *   <li>LLAMA_SPLIT_MODE_NONE: Use single GPU specified by main_gpu</li>
     *   <li>LLAMA_SPLIT_MODE_LAYER: Split layers across GPUs</li>
     *   <li>LLAMA_SPLIT_MODE_ROW: Split individual layers by rows</li>
     * </ul>
     */
    public int split_mode = 0;

    /**
     * Primary GPU device ID to use when split_mode is LLAMA_SPLIT_MODE_NONE.
     * <p>
     * Specifies which GPU device should handle the entire model when not
     * using multi-GPU splitting. Also serves as the primary GPU for operations
     * that require a single device even in multi-GPU configurations.
     */
    public int main_gpu = 0;

    /**
     * Proportion of model (layers or rows) to offload to each GPU.
     * <p>
     * Array of float values controlling how much of the model goes to each GPU.
     * Array size must equal {@code llama_max_devices()}. Values represent
     * proportions (0.0 to 1.0) for distributing model components.
     * <p>
     * <strong>Usage:</strong>
     * <ul>
     *   <li>Layer mode: Proportion of layers per GPU</li>
     *   <li>Row mode: Proportion of rows per layer per GPU</li>
     * </ul>
     * Type: {@code const float*}
     */
    public Pointer tensor_split = null;

    /**
     * Progress callback function called during model loading.
     * <p>
     * Called with progress values between 0.0 and 1.0 to report loading progress.
     * If the callback returns true, loading continues. If false, loading is aborted.
     * Pass NULL to disable progress reporting.
     * <p>
     * <strong>Signature:</strong> {@code bool (*llama_progress_callback)(float progress, void* ctx)}
     * Type: {@code llama_progress_callback}
     */
    public Pointer progress_callback = null;

    /**
     * User data pointer passed to the progress callback function.
     * <p>
     * This pointer is passed as the second argument to the progress_callback
     * function, allowing the callback to access application-specific context
     * or state during model loading.
     * Type: {@code void*}
     */
    public Pointer progress_callback_user_data = null;

    /**
     * Array of key-value overrides for model metadata.
     * <p>
     * Allows overriding specific metadata values in the model file.
     * Useful for customizing model behavior without modifying the original file.
     * Must be a NULL-terminated array of {@code llama_model_kv_override} structures.
     * <p>
     * Type: {@code const struct llama_model_kv_override*}
     */
    public Pointer kv_overrides = null;

    private transient LlamaModelTensorBuftOverrideNative[] tensorBuftOverridesCache;
    private transient LlamaModelKvOverrideNative[] kvOverridesCache;

    //
    // Boolean Flags
    // Keep the booleans together to avoid misalignment during copy-by-value.
    // Use byte instead of boolean to match C bool size exactly.
    //
    
    /**
     * Load only the model vocabulary, skip loading weights.
     * <p>
     * When enabled, only tokenization capabilities are available.
     * Useful for applications that only need text tokenization without
     * inference capabilities. Significantly reduces memory usage.
     */
    public byte vocab_only = 0;
    
    /**
     * Use memory mapping (mmap) for model file access if possible.
     * <p>
     * Memory mapping can improve loading performance and reduce memory usage
     * by allowing the OS to manage model data paging. May not be available
     * on all platforms or file systems.
     * <p>
     * <strong>Default:</strong> Enabled (1)
     */
    public byte use_mmap = 1;
    
    /**
     * Force the system to keep model data in RAM (memory locking).
     * <p>
     * Prevents model data from being swapped to disk, ensuring consistent
     * performance at the cost of increased memory pressure. Requires
     * appropriate system permissions on some platforms.
     * <p>
     * <strong>Default:</strong> Disabled (0)
     */
    public byte use_mlock = 0;
    
    /**
     * Validate model tensor data during loading.
     * <p>
     * Performs integrity checks on tensor data to detect corruption
     * or invalid model files. Adds loading time but provides better
     * error detection for debugging.
     * <p>
     * <strong>Default:</strong> Enabled (1)
     */
    public byte check_tensors = 1;
    
    /**
     * Use extra buffer types for advanced memory optimization.
     * <p>
     * Enables additional buffer allocation strategies used for weight
     * repacking and memory layout optimization. May improve performance
     * in specific scenarios but increases complexity.
     * <p>
     * <strong>Default:</strong> Disabled (0)
     */
    public byte use_extra_bufts = 0;

    /**
     * Defines the order of fields in the native structure.
     * <p>
     * This order must match exactly with the C structure definition
     * to ensure proper memory layout and JNA marshalling.
     */
    @Override
    protected java.util.List<String> getFieldOrder() {
        return java.util.Arrays.asList("devices", "tensor_buft_overrides", "n_gpu_layers", "split_mode",
                "main_gpu", "tensor_split", "progress_callback", "progress_callback_user_data",
                "kv_overrides", "vocab_only", "use_mmap", "use_mlock", "check_tensors", "use_extra_bufts");
    }
    
    /**
     * Creates a new LlamaModelParamsNative instance with default values.
     * <p>
     * Default configuration:
     * <ul>
     *   <li>n_gpu_layers = 0 (CPU-only)</li>
     *   <li>split_mode = 0 (LLAMA_SPLIT_MODE_NONE)</li>
     *   <li>main_gpu = 0 (first GPU device)</li>
     *   <li>vocab_only = 0 (load full model)</li>
     *   <li>use_mmap = 1 (memory mapping enabled)</li>
     *   <li>use_mlock = 0 (memory locking disabled)</li>
     *   <li>check_tensors = 1 (tensor validation enabled)</li>
     *   <li>use_extra_bufts = 0 (extra buffers disabled)</li>
     *   <li>All pointers = null</li>
     * </ul>
     */
    public LlamaModelParamsNative() {
        super();
    }
    
    /**
     * Creates a new LlamaModelParamsNative instance from a native pointer.
     * <p>
     * This constructor is used when the structure is returned from
     * native functions or when working with existing native memory.
     * The structure is automatically read from the native memory.
     * 
     * @param peer pointer to existing native structure
     */
    public LlamaModelParamsNative(Pointer peer) {
        super(peer);
        read();
    }

    public void setTensorBuftOverrides(LlamaModelTensorBuftOverride... overrides) {
        if (overrides == null || overrides.length == 0) {
            tensor_buft_overrides = Pointer.NULL;
            tensorBuftOverridesCache = null;
            return;
        }

        LlamaModelTensorBuftOverrideNative[] natives = new LlamaModelTensorBuftOverrideNative[overrides.length + 1];
        for (int i = 0; i < overrides.length; i++) {
            natives[i] = overrides[i].toNative();
        }
        natives[natives.length - 1] = new LlamaModelTensorBuftOverrideNative();
        natives[natives.length - 1].bufferType = new GgmlBackendBufferTypeNative(Pointer.NULL);
        natives[natives.length - 1].write();

        tensorBuftOverridesCache = natives;
        tensor_buft_overrides = natives[0].getPointer();
    }

    public void setKvOverrides(LlamaModelKvOverride... overrides) {
        if (overrides == null || overrides.length == 0) {
            kv_overrides = Pointer.NULL;
            kvOverridesCache = null;
            return;
        }

        LlamaModelKvOverrideNative[] natives = new LlamaModelKvOverrideNative[overrides.length + 1];
        for (int i = 0; i < overrides.length; i++) {
            natives[i] = overrides[i].toNative();
        }
        natives[natives.length - 1] = new LlamaModelKvOverrideNative();
        natives[natives.length - 1].setKey(null);
        natives[natives.length - 1].write();

        kvOverridesCache = natives;
        kv_overrides = natives[0].getPointer();
    }
}
