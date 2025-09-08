package com.quasarbyte.llama.cpp.jna.library.declaration.llama;

import com.quasarbyte.llama.cpp.jna.library.declaration.LlamaCppBase;
import com.quasarbyte.llama.cpp.jna.library.declaration.UInt32;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.sun.jna.Callback;
import com.sun.jna.Pointer;

/**
 * Java Native Access (JNA) interface for llama.cpp C library.
 * <p>
 * This interface provides direct bindings to the llama.cpp C API for loading,
 * running, and managing LLaMA language models. It corresponds to the functions
 * and constants defined in llama.h from the llama.cpp project.
 * <p>
 * All functions use explicit C calling convention via Function.C_CONVENTION for
 * proper native library interoperability.
 * <p>
 * <strong>Usage Pattern:</strong>
 * <ol>
 *   <li>Initialize backend with {@link #llama_backend_init()}</li>
 *   <li>Load model with {@link #llama_model_load_from_file(String, LlamaModelParamsNative)}</li>
 *   <li>Create context with {@link #llama_init_from_model(LlamaModelNative, LlamaContextParamsNative)}</li>
 *   <li>Process text with {@link #llama_decode(LlamaContextNative, LlamaBatchNative)}</li>
 *   <li>Clean up resources in reverse order</li>
 *   <li>Free backend with {@link #llama_backend_free()}</li>
 * </ol>
 *
 * @see <a href="https://github.com/ggml-org/llama.cpp">llama.cpp GitHub Repository</a>
 */
public interface LlamaLibrary extends LlamaCppBase<LlamaLibrary> {

    //
    // Log Level Constants (from ggml.h)
    //

    /**
     * No logging
     */
    int GGML_LOG_LEVEL_NONE  = 0;

    /**
     * Debug level logging
     */
    int GGML_LOG_LEVEL_DEBUG = 1;

    /**
     * Info level logging
     */
    int GGML_LOG_LEVEL_INFO  = 2;

    /**
     * Warning level logging
     */
    int GGML_LOG_LEVEL_WARN  = 3;

    /**
     * Error level logging
     */
    int GGML_LOG_LEVEL_ERROR = 4;

    /**
     * Continue previous log message
     */
    int GGML_LOG_LEVEL_CONT  = 5;

    //
    // Callback Interfaces
    //

    /**
     * Callback interface for llama.cpp log messages.
     * Matches the C signature: void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)
     */
    interface LlamaLogCallback extends Callback {
        /**
         * Called when llama.cpp wants to log a message
         *
         * @param level Log level (GGML_LOG_LEVEL_*)
         * @param text Log message text
         * @param userData User data pointer (can be null)
         */
        void invoke(int level, String text, Pointer userData);
    }

    //
    // Core Constants
    //

    /**
     * Invalid/null token identifier
     */
    int LLAMA_TOKEN_NULL = -1;

    /**
     * Default seed value for random number generation (0xFFFFFFFF)
     */
    long LLAMA_DEFAULT_SEED = 0xFFFFFFFFL;

    //
    // File Format Magic Numbers
    //

    /**
     * GGLA file magic number ('ggla' in ASCII)
     */
    int LLAMA_FILE_MAGIC_GGLA = 0x67676c61;

    /**
     * GGSN file magic number ('ggsn' in ASCII)
     */
    int LLAMA_FILE_MAGIC_GGSN = 0x6767736e;

    /**
     * GGSQ file magic number ('ggsq' in ASCII)
     */
    int LLAMA_FILE_MAGIC_GGSQ = 0x67677371;

    //
    // Session File Format
    //

    /**
     * Session file magic number (same as GGSN)
     */
    int LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN;

    /**
     * Session file format version
     */
    int LLAMA_SESSION_VERSION = 9;

    //
    // State Sequence Format
    //

    /**
     * State sequence file magic number (same as GGSQ)
     */
    int LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ;

    /**
     * State sequence file format version
     */
    int LLAMA_STATE_SEQ_VERSION = 2;

    /**
     * Flag for SWA (Sliding Window Attention) only state sequences
     */
    int LLAMA_STATE_SEQ_FLAGS_SWA_ONLY = 1;

    //
    // Backend Initialization
    //

    /**
     * Initialize the llama + ggml backend.
     * <p>
     *  The comment in llama.h (“Call once at the start of the program”) isn’t just decoration: llama_backend_init() wires up the backend registry, sets NUMA/thread‑pool state, and gives freshly loaded plugins a chance to register accelerators.
     *  When the C++ examples omit it they get away with it because the runtime lazily initialises on first use, but you lose deterministic control and GPU plugins may never activate (they will happily fall back to CPU).
     * <p>
     * Call this once after loading the GGML plugins you plan to use (for example via
     * {@code ggml_backend_load_all_from_path(...)}) and before creating models or contexts.
     * The plain {@code ggml_backend_load_all()} only scans the executable and current directories;
     * if plugins live elsewhere you must load them explicitly or point {@code GGML_BACKEND_PATH}
     * at the concrete library files. Once plugins are present this call wires up the registry,
     * NUMA state, and thread pools used during inference.
     */
    void llama_backend_init();

    /**
     * Free the llama + ggml backend resources.
     * <p>
     * Pair this with {@link #llama_backend_init()} once you are finished using the library.
     * In long-running JVMs call it after unloading models so GPU/CPU plugins can release
     * device resources that were registered during initialization.
     */
    void llama_backend_free();

    /**
     * Initialize NUMA (Non-Uniform Memory Access) optimizations.
     * <p>
     * This is optional and should be called after {@link #llama_backend_init()}
     * if you want to enable NUMA optimizations for better performance on
     * multi-socket systems.
     *
     * @param numa NUMA strategy from ggml_numa_strategy enum
     */
    void llama_numa_init(int numa);

    //
    // System Capabilities
    //

    /**
     * Check if memory mapping (mmap) is supported on this system
     */
    boolean llama_supports_mmap();

    /**
     * Check if memory locking (mlock) is supported on this system
     */
    boolean llama_supports_mlock();

    /**
     * Check if GPU offloading is supported on this system
     */
    boolean llama_supports_gpu_offload();

    /**
     * Check if RPC (Remote Procedure Call) is supported on this system
     */
    boolean llama_supports_rpc();

    /**
     * Get the maximum number of devices available for model distribution
     */
    long llama_max_devices();

    /**
     * Get the maximum number of parallel sequences supported
     */
    long llama_max_parallel_sequences();

    //
    // Timing Functions
    //

    /**
     * Get current time in microseconds
     */
    long llama_time_us();

    //
    // Thread Pool Management
    //

    /**
     * Attach custom thread pools to a context.
     * <p>
     * An auto threadpool gets created in ggml if not passed explicitly.
     *
     * @param ctx              the llama context
     * @param threadpool       thread pool for generation (single token)
     * @param threadpool_batch thread pool for batch processing (multiple tokens)
     */
    void llama_attach_threadpool(LlamaContextNative ctx, LlamaThreadPoolNative threadpool, LlamaThreadPoolBatchNative threadpool_batch);

    /**
     * Detach thread pools from a context.
     *
     * @param ctx the llama context
     */
    void llama_detach_threadpool(LlamaContextNative ctx);

    //
    // System Information
    //

    /**
     * Get system information as a formatted string
     */
    String llama_print_system_info();

    //
    // Logging
    //

    /**
     * Set callback for all future logging events.
     * <p>
     * If this is not called, or NULL is supplied, everything is output on stderr.
     *
     * @param log_callback callback function for log messages (or null to disable)
     * @param user_data    user data passed to the callback (or null)
     */
    void llama_log_set(LlamaLogCallback log_callback, Pointer user_data);

    /**
     * Get the name of a flash attention type.
     *
     * @param flash_attn_type flash attention type enum value
     * @return human-readable name of the flash attention type
     */
    String llama_flash_attn_type_name(int flash_attn_type);

    //
    // Default Parameter Functions
    //
    // These functions return structures by value.
    // JNA will handle the structure marshalling automatically.
    // TODO: update API to start accepting pointers to params structs
    //

    /**
     * Get default model parameters with recommended settings
     */
    LlamaModelParamsNative llama_model_default_params();

    /**
     * Get default context parameters with recommended settings
     */
    LlamaContextParamsNative llama_context_default_params();

    /**
     * Get default sampler chain parameters with recommended settings
     */
    LlamaSamplerChainParamsNative llama_sampler_chain_default_params();

    //
    // Model Loading and Management
    //

    /**
     * Load a model from a single file.
     * <p>
     * If the file is split into multiple parts, the file name must follow this pattern:
     * {@code <name>-%05d-of-%05d.gguf}
     * <p>
     * If the split file name does not follow this pattern, use
     * {@link #llama_model_load_from_splits(String[], long, LlamaModelParamsNative)} instead.
     *
     * @param path_model path to the model file
     * @param params     model loading parameters
     * @return pointer to the loaded model, or null on failure
     */
    LlamaModelNative llama_model_load_from_file(String path_model, LlamaModelParamsNative params);

    /**
     * Load a model from multiple split files (supports custom naming scheme).
     * <p>
     * The paths must be provided in the correct order.
     *
     * @param paths   array of file paths in correct order
     * @param n_paths number of files in the array
     * @param params  model loading parameters
     * @return pointer to the loaded model, or null on failure
     */
    LlamaModelNative llama_model_load_from_splits(String[] paths, long n_paths, LlamaModelParamsNative params);

    /**
     * Save a model to a file.
     *
     * @param model      pointer to the model
     * @param path_model path where to save the model
     */
    void llama_model_save_to_file(LlamaModelNative model, String path_model);

    /**
     * Free a model and release all associated resources.
     * <p>
     * This should be called for every model loaded with
     * {@link #llama_model_load_from_file} or {@link #llama_model_load_from_splits}.
     *
     * @param model pointer to the model to freeSampler
     */
    void llama_model_free(LlamaModelNative model);

    //
    // Model Information
    //

    /**
     * Get the vocabulary associated with a model
     */
    LlamaVocabularyNative llama_model_get_vocab(LlamaModelNative model);

    /**
     * Get the RoPE type used by this model
     */
    int llama_model_rope_type(LlamaModelNative model);

    /**
     * Get the size of the context used during training
     */
    int llama_model_n_ctx_train(LlamaModelNative model);

    /**
     * Get the embedding dimension of the model
     */
    int llama_model_n_embd(LlamaModelNative model);

    /**
     * Get the number of layers in the model
     */
    int llama_model_n_layer(LlamaModelNative model);

    /**
     * Get the number of attention heads in the model
     */
    int llama_model_n_head(LlamaModelNative model);

    /**
     * Get the number of key-value heads in the model (for grouped-query attention)
     */
    int llama_model_n_head_kv(LlamaModelNative model);

    /**
     * Get the number of SWA (Sliding Window Attention) layers
     */
    int llama_model_n_swa(LlamaModelNative model);

    /**
     * Get the RoPE frequency scaling factor used during training
     */
    float llama_model_rope_freq_scale_train(LlamaModelNative model);

    /**
     * Get the number of classification output classes
     */
    int llama_model_n_cls_out(LlamaModelNative model);

    /**
     * Get classification label for a given class index.
     *
     * @param model model pointer
     * @param i     class index
     * @return classification label string
     */
    String llama_model_cls_label(LlamaModelNative model, UInt32 i);

    /**
     * Get model metadata value as string.
     *
     * @param model    model pointer
     * @param key      metadata key
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the string, or negative on error
     */
    int llama_model_meta_val_str(LlamaModelNative model, String key, byte[] buf, long buf_size);

    /**
     * Get the number of metadata key-value pairs in the model
     */
    int llama_model_meta_count(LlamaModelNative model);

    /**
     * Get metadata key by index.
     *
     * @param model    model pointer
     * @param i        metadata index
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the key string, or negative on error
     */
    int llama_model_meta_key_by_index(LlamaModelNative model, int i, byte[] buf, long buf_size);

    /**
     * Get metadata value by index.
     *
     * @param model    model pointer
     * @param i        metadata index
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the value string, or negative on error
     */
    int llama_model_meta_val_str_by_index(LlamaModelNative model, int i, byte[] buf, long buf_size);

    /**
     * Get model description string.
     *
     * @param model    model pointer
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the description, or negative on error
     */
    int llama_model_desc(LlamaModelNative model, byte[] buf, long buf_size);

    /**
     * Get the total size of the model in bytes
     */
    long llama_model_size(LlamaModelNative model);

    /**
     * Get chat template for the model.
     *
     * @param model model pointer
     * @param name  template name (can be null for default)
     * @return chat template string, or null if not found
     */
    String llama_model_chat_template(LlamaModelNative model, String name);

    /**
     * Get the total number of parameters in the model
     */
    long llama_model_n_params(LlamaModelNative model);

    /**
     * Check if the model has an encoder component
     */
    boolean llama_model_has_encoder(LlamaModelNative model);

    /**
     * Check if the model has a decoder component
     */
    boolean llama_model_has_decoder(LlamaModelNative model);

    /**
     * Get the decoder start token for encoder-decoder models
     */
    int llama_model_decoder_start_token(LlamaModelNative model);

    /**
     * Check if the model uses recurrent architecture
     */
    boolean llama_model_is_recurrent(LlamaModelNative model);

    /**
     * Check if the model is a diffusion model
     */
    boolean llama_model_is_diffusion(LlamaModelNative model);

    /**
     * Quantize a model file.
     *
     * @param fname_inp input model file path
     * @param fname_out output quantized model file path
     * @param params    quantization parameters
     * @return 0 on success, non-zero on error
     */
    UInt32 llama_model_quantize(String fname_inp, String fname_out, LlamaModelQuantizeParamsNative params);

    //
    // Context Functions
    //

    /**
     * Create a new context from a model.
     * <p>
     * This function creates an inference context that can be used to generate text,
     * process batches, and manage the model's state during inference.
     *
     * @param model  pointer to the loaded model
     * @param params context parameters
     * @return pointer to the new context, or null on failure
     */
    LlamaContextNative llama_init_from_model(LlamaModelNative model, LlamaContextParamsNative params);

    /**
     * Free a context and release all associated resources.
     * <p>
     * This should be called for every context created with {@link #llama_init_from_model}.
     *
     * @param ctx pointer to the context to freeSampler
     */
    void llama_free(LlamaContextNative ctx);

    /**
     * Get the size of the context (number of tokens)
     */
    long llama_n_ctx(LlamaContextNative ctx);

    /**
     * Get the batch size for this context
     */
    long llama_n_batch(LlamaContextNative ctx);

    /**
     * Get the micro-batch size for this context
     */
    long llama_n_ubatch(LlamaContextNative ctx);

    /**
     * Get the maximum number of sequences this context can handle
     */
    long llama_n_seq_max(LlamaContextNative ctx);

    /**
     * Get the model associated with this context
     */
    LlamaModelNative llama_get_model(LlamaContextNative ctx);

    /**
     * Get the memory manager associated with this context
     */
    LlamaMemoryManagerNative llama_get_memory(LlamaContextNative ctx);

    /**
     * Get the pooling type for this context
     */
    int llama_pooling_type(LlamaContextNative ctx);

    //
    // Context Configuration
    //

    /**
     * Set the number of threads for single-token and batch processing.
     *
     * @param ctx             context pointer
     * @param n_threads       number of threads for single-token processing
     * @param n_threads_batch number of threads for batch processing
     */
    void llama_set_n_threads(LlamaContextNative ctx, int n_threads, int n_threads_batch);

    /**
     * Get the number of threads used for single-token processing
     */
    int llama_n_threads(LlamaContextNative ctx);

    /**
     * Get the number of threads used for batch processing
     */
    int llama_n_threads_batch(LlamaContextNative ctx);

    /**
     * Enable or disable embeddings output.
     *
     * @param ctx        context pointer
     * @param embeddings true to enable embeddings, false to disable
     */
    void llama_set_embeddings(LlamaContextNative ctx, boolean embeddings);

    /**
     * Set causal attention mode.
     *
     * @param ctx         context pointer
     * @param causal_attn true for causal attention, false for non-causal
     */
    void llama_set_causal_attn(LlamaContextNative ctx, boolean causal_attn);

    /**
     * Enable or disable warmup mode.
     *
     * @param ctx    context pointer
     * @param warmup true to enable warmup, false to disable
     */
    void llama_set_warmup(LlamaContextNative ctx, boolean warmup);

    /**
     * Set abort callback function.
     * <p>
     * If the callback returns true, execution of llama_decode() will be aborted.
     * Currently works only with CPU execution.
     *
     * @param ctx                 context pointer
     * @param abort_callback      callback function pointer
     * @param abort_callback_data user data passed to callback
     */
    void llama_set_abort_callback(LlamaContextNative ctx, Pointer abort_callback, Pointer abort_callback_data);

    /**
     * Wait for all operations on the context to complete.
     *
     * @param ctx context pointer
     */
    void llama_synchronize(LlamaContextNative ctx);

    //
    // Vocabulary Functions
    //

    /**
     * Get the type of vocabulary used by the model
     */
    int llama_vocab_type(LlamaVocabularyNative vocab);

    /**
     * Get the total number of tokens in the vocabulary
     */
    int llama_vocab_n_tokens(LlamaVocabularyNative vocab);

    //
    // Token Information Functions
    //

    /**
     * Get the text representation of a token
     */
    String llama_vocab_get_text(LlamaVocabularyNative vocab, int token);

    /**
     * Get the score/probability of a token
     */
    float llama_vocab_get_score(LlamaVocabularyNative vocab, int token);

    /**
     * Get the attributes of a token (bitfield of llama_token_attr)
     */
    int llama_vocab_get_attr(LlamaVocabularyNative vocab, int token);

    /**
     * Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
     */
    boolean llama_vocab_is_eog(LlamaVocabularyNative vocab, int token);

    /**
     * Identify if Token Id is a control token or a render-able token
     */
    boolean llama_vocab_is_control(LlamaVocabularyNative vocab, int token);

    //
    // Special Tokens
    //

    /**
     * Get beginning-of-sentence token ID
     */
    int llama_vocab_bos(LlamaVocabularyNative vocab);

    /**
     * Get end-of-sentence token ID
     */
    int llama_vocab_eos(LlamaVocabularyNative vocab);

    /**
     * Get end-of-turn token ID
     */
    int llama_vocab_eot(LlamaVocabularyNative vocab);

    /**
     * Get sentence separator token ID
     */
    int llama_vocab_sep(LlamaVocabularyNative vocab);

    /**
     * Get next-line token ID
     */
    int llama_vocab_nl(LlamaVocabularyNative vocab);

    /**
     * Get padding token ID
     */
    int llama_vocab_pad(LlamaVocabularyNative vocab);

    /**
     * Get mask token ID
     */
    int llama_vocab_mask(LlamaVocabularyNative vocab);

    /**
     * Check if BOS token should be added automatically
     */
    boolean llama_vocab_get_add_bos(LlamaVocabularyNative vocab);

    /**
     * Check if EOS token should be added automatically
     */
    boolean llama_vocab_get_add_eos(LlamaVocabularyNative vocab);

    /**
     * Check if separator token should be added automatically
     */
    boolean llama_vocab_get_add_sep(LlamaVocabularyNative vocab);

    //
    // Fill-In-Middle (FIM) Tokens
    //

    /**
     * Get FIM prefix token ID
     */
    int llama_vocab_fim_pre(LlamaVocabularyNative vocab);

    /**
     * Get FIM suffix token ID
     */
    int llama_vocab_fim_suf(LlamaVocabularyNative vocab);

    /**
     * Get FIM middle token ID
     */
    int llama_vocab_fim_mid(LlamaVocabularyNative vocab);

    /**
     * Get FIM padding token ID
     */
    int llama_vocab_fim_pad(LlamaVocabularyNative vocab);

    /**
     * Get FIM repository token ID
     */
    int llama_vocab_fim_rep(LlamaVocabularyNative vocab);

    /**
     * Get FIM separator token ID
     */
    int llama_vocab_fim_sep(LlamaVocabularyNative vocab);

    //
    // Tokenization Functions
    //
    // The API is thread-safe.
    //

    /**
     * Convert the provided text into tokens.
     *
     * @param vocab         vocabulary pointer
     * @param text          input text to tokenize
     * @param text_len      length of input text (-1 for null-terminated)
     * @param tokens        output array for tokens (must be large enough)
     * @param n_max_tokens  maximum number of tokens to write
     * @param add_special   allow adding BOS and EOS tokens if model is configured to do so
     * @param parse_special allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does not insert a leading space.
     * @return number of tokens on success, negative number on failure
     * Returns a negative number on failure - the number of tokens that would have been returned
     * Returns INT32_MIN on overflow (tokenization result size exceeds int32_t limit)
     */
    int llama_tokenize(LlamaVocabularyNative vocab, byte[] text, int text_len, Pointer tokens, int n_max_tokens, boolean add_special, boolean parse_special);

    /**
     * Convert token ID to text piece.
     * <p>
     * Uses the vocabulary in the provided context.
     * Does not write null terminator to the buffer.
     *
     * @param vocab   vocabulary pointer
     * @param token   token ID to convert
     * @param buf     output buffer for text
     * @param length  buffer size
     * @param lstrip  number of leading spaces to skip before copying
     * @param special if true, special tokens are rendered in the output
     * @return number of bytes written, or negative on error
     */
    int llama_token_to_piece(LlamaVocabularyNative vocab, int token, byte[] buf, int length, int lstrip, boolean special);

    /**
     * Convert tokens back to text (inverse of llama_tokenize).
     *
     * @param vocab           vocabulary pointer
     * @param tokens          array of tokens to convert
     * @param n_tokens        number of tokens in array
     * @param text            output buffer for text
     * @param text_len_max    maximum size of output buffer
     * @param remove_special  if true, remove special tokens from output
     * @param unparse_special if true, render special tokens as text
     * @return number of chars/bytes on success, negative on failure
     */
    int llama_detokenize(LlamaVocabularyNative vocab, Pointer tokens, int n_tokens, byte[] text, int text_len_max, boolean remove_special, boolean unparse_special);

    //
    // Chat Template Functions
    //

    /**
     * Apply chat template to format conversation messages.
     *
     * @param tmpl    chat template string
     * @param chat    pointer to chat messages array
     * @param n_msg   number of messages in the chat
     * @param add_ass whether to add assistant message
     * @param buf     output buffer for formatted chat
     * @param length  buffer size
     * @return length of the formatted chat, or negative on error
     */
    int llama_chat_apply_template(String tmpl, LlamaChatMessageNative[] chat, int n_msg, boolean add_ass, byte[] buf, int length);

    /**
     * Get list of built-in chat templates.
     *
     * @param output output buffer for template names
     * @param len    buffer size
     * @return number of templates, or negative on error
     */
    int llama_chat_builtin_templates(String[] output, long len);

    //
    // Batch Functions
    //
    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    //

    /**
     * Create a simple batch with tokens for single sequence processing.
     *
     * @param tokens   pointer to token array
     * @param n_tokens number of tokens
     * @return initialized batch structure
     */
    LlamaBatchNative llama_batch_get_one(Pointer tokens, int n_tokens);

    /**
     * Initialize a batch structure for multi-sequence processing.
     *
     * @param n_tokens  maximum number of tokens
     * @param embd      embedding dimension (0 if using tokens)
     * @param n_seq_max maximum number of sequences
     * @return initialized batch structure
     */
    LlamaBatchNative llama_batch_init(int n_tokens, int embd, int n_seq_max);

    /**
     * Free resources allocated for a batch.
     *
     * @param batch the batch to freeSampler
     */
    void llama_batch_free(LlamaBatchNative batch);

    //
    // Processing Functions
    //

    /**
     * Encode input batch (typically for embeddings).
     *
     * @param ctx   context pointer
     * @param batch input batch
     * @return 0 on success, positive on warning, negative on error
     */
    int llama_encode(LlamaContextNative ctx, LlamaBatchNative batch);

    /**
     * Decode input batch (for text generation).
     * <p>
     * Processes the batch and updates the model's internal state.
     * After calling this, use llama_get_logits() to get the output probabilities.
     *
     * @param ctx   context pointer
     * @param batch input batch
     * @return 0 on success, positive on warning, negative on error
     */
    int llama_decode(LlamaContextNative ctx, LlamaBatchNative batch);

    //
    // Result Access Functions
    //

    /**
     * Get output logits for the last token in the last sequence.
     *
     * @param ctx context pointer
     * @return pointer to logits array, or null if no logits available
     */
    Pointer llama_get_logits(LlamaContextNative ctx);

    /**
     * Get output logits for a specific token position.
     *
     * @param ctx context pointer
     * @param i   token position index
     * @return pointer to logits array for the specified position
     */
    Pointer llama_get_logits_ith(LlamaContextNative ctx, int i);

    /**
     * Get embeddings for the last token in the last sequence.
     *
     * @param ctx context pointer
     * @return pointer to embeddings array, or null if no embeddings available
     */
    Pointer llama_get_embeddings(LlamaContextNative ctx);

    /**
     * Get embeddings for a specific token position.
     *
     * @param ctx context pointer
     * @param i   token position index
     * @return pointer to embeddings array for the specified position
     */
    Pointer llama_get_embeddings_ith(LlamaContextNative ctx, int i);

    /**
     * Get embeddings for a specific sequence.
     *
     * @param ctx    context pointer
     * @param seq_id sequence ID
     * @return pointer to embeddings array for the specified sequence
     */
    Pointer llama_get_embeddings_seq(LlamaContextNative ctx, int seq_id);

    //
    // Sampler Functions
    //

    /**
     * Initialize a custom sampler with interface and context
     */
    LlamaSamplerNative llama_sampler_init(Pointer iface, LlamaContextNative ctx);

    /**
     * Get the name of a sampler
     */
    String llama_sampler_name(LlamaSamplerNative smpl);

    /**
     * Accept a token and update sampler state
     */
    void llama_sampler_accept(LlamaSamplerNative smpl, int token);

    /**
     * Apply sampler to modify token probabilities
     */
    void llama_sampler_apply(LlamaSamplerNative smpl, LlamaTokenDataArrayNative cur_p);

    /**
     * Reset sampler to initial state
     */
    void llama_sampler_reset(LlamaSamplerNative smpl);

    /**
     * Clone a sampler (deep copy)
     */
    LlamaSamplerNative llama_sampler_clone(LlamaSamplerNative smpl);

    /**
     * Free sampler resources
     */
    void llama_sampler_free(LlamaSamplerNative smpl);

    //
    // Sampler Chain Functions
    //

    /**
     * Initialize a sampler chain for combining multiple samplers
     */
    LlamaSamplerNative llama_sampler_chain_init(LlamaSamplerChainParamsNative params);

    /**
     * Add a sampler to the chain
     */
    void llama_sampler_chain_add(LlamaSamplerNative chain, LlamaSamplerNative smpl);

    /**
     * Get sampler at specific index in the chain
     */
    LlamaSamplerNative llama_sampler_chain_get(LlamaSamplerNative chain, int i);

    /**
     * Get number of samplers in the chain
     */
    int llama_sampler_chain_n(LlamaSamplerNative chain);

    /**
     * Remove and return sampler at specific index
     */
    LlamaSamplerNative llama_sampler_chain_remove(LlamaSamplerNative chain, int i);

    //
    // Individual Sampler Initializers
    //

    /**
     * Initialize greedy sampler (always picks highest probability token)
     */
    LlamaSamplerNative llama_sampler_init_greedy();

    /**
     * Initialize distribution sampler (samples from the full distribution).
     *
     * @param seed random seed for sampling
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_dist(UInt32 seed);

    /**
     * Initialize top-k sampler (only consider top k tokens).
     *
     * @param k number of top tokens to consider
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_top_k(int k);

    /**
     * Initialize top-p (nucleus) sampler.
     *
     * @param p        cumulative probability threshold (0.0 to 1.0)
     * @param min_keep minimum number of tokens to keep
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_top_p(float p, long min_keep);

    /**
     * Initialize min-p sampler (minimum probability threshold).
     *
     * @param p        minimum probability threshold relative to the most likely token
     * @param min_keep minimum number of tokens to keep
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_min_p(float p, long min_keep);

    /**
     * Initialize typical sampling (locally typical sampling).
     *
     * @param p        typical sampling parameter
     * @param min_keep minimum number of tokens to keep
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_typical(float p, long min_keep);

    /**
     * Initialize temperature sampler.
     *
     * @param t temperature value (higher = more random, lower = more focused)
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_temp(float t);

    /**
     * Initialize extended temperature sampler with additional parameters.
     *
     * @param t        base temperature
     * @param delta    temperature delta
     * @param exponent temperature exponent
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_temp_ext(float t, float delta, float exponent);

    /**
     * Initialize XTC (eXtended Temperature Control) sampler.
     *
     * @param p        probability threshold
     * @param t        temperature
     * @param min_keep minimum number of tokens to keep
     * @param seed     random seed
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_xtc(float p, float t, long min_keep, UInt32 seed);

    /**
     * Initialize top-n-sigma sampler.
     *
     * @param n sigma parameter
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_top_n_sigma(float n);

    /**
     * Initialize Mirostat v1 sampler.
     *
     * @param n_vocab vocabulary size
     * @param seed    random seed
     * @param tau     target surprise value
     * @param eta     learning rate
     * @param m       number of candidates to consider
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_mirostat(int n_vocab, int seed, float tau, float eta, UInt32 m);

    /**
     * Initialize Mirostat v2 sampler.
     *
     * @param seed random seed
     * @param tau  target surprise value
     * @param eta  learning rate
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_mirostat_v2(UInt32 seed, float tau, float eta);

    /**
     * Initialize grammar-guided sampler.
     *
     * @param vocab        vocabulary pointer
     * @param grammar_str  grammar string
     * @param grammar_root root rule name
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_grammar(LlamaVocabularyNative vocab, String grammar_str, String grammar_root);

    /**
     * Initialize lazy grammar sampler with pattern triggers.
     *
     * @param vocab                vocabulary pointer
     * @param grammar_str          grammar string
     * @param grammar_root         root rule name
     * @param trigger_patterns     trigger pattern array
     * @param num_trigger_patterns number of trigger patterns
     * @param trigger_tokens       trigger token array
     * @param num_trigger_tokens   number of trigger tokens
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_grammar_lazy_patterns(LlamaVocabularyNative vocab, String grammar_str, String grammar_root, String[] trigger_patterns, long num_trigger_patterns, int[] trigger_tokens, long num_trigger_tokens);

    /**
     * Initialize repetition penalty sampler.
     *
     * @param penalty_last_n  number of last tokens to consider for penalties
     * @param penalty_repeat  repetition penalty multiplier
     * @param penalty_freq    frequency penalty multiplier
     * @param penalty_present presence penalty multiplier
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_penalties(int penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);

    /**
     * Initialize DRY (Don't Repeat Yourself) sampler.
     *
     * @param vocab              vocabulary pointer
     * @param n_ctx_train        training context size
     * @param dry_multiplier     DRY penalty multiplier
     * @param dry_base           DRY base value
     * @param dry_allowed_length allowed repetition length
     * @param dry_penalty_last_n number of tokens to consider
     * @param seq_breakers       sequence breaker tokens
     * @param num_breakers       number of sequence breakers
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_dry(LlamaVocabularyNative vocab, int n_ctx_train, float dry_multiplier, float dry_base, int dry_allowed_length, int dry_penalty_last_n, String[] seq_breakers, long num_breakers);

    /**
     * Initialize logit bias sampler.
     *
     * @param n_vocab      vocabulary size
     * @param n_logit_bias number of biased tokens
     * @param logit_bias   array of bias values
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_logit_bias(int n_vocab, int n_logit_bias, LlamaLogitBiasNative[] logit_bias);

    /**
     * Initialize infill sampler for fill-in-the-middle tasks.
     *
     * @param vocab vocabulary pointer
     * @return sampler pointer
     */
    LlamaSamplerNative llama_sampler_init_infill(LlamaVocabularyNative vocab);

    /**
     * Get the seed used by a sampler.
     *
     * @param smpl sampler pointer
     * @return seed value, or -1 if not applicable
     */
    UInt32 llama_sampler_get_seed(LlamaSamplerNative smpl);

    /**
     * Sample a token using the sampler.
     *
     * @param smpl sampler pointer
     * @param ctx  context pointer
     * @param idx  index in the logits array (-1 for last)
     * @return sampled token ID
     */
    int llama_sampler_sample(LlamaSamplerNative smpl, LlamaContextNative ctx, int idx);

    //
    // Memory Management Functions
    //

    /**
     * Clear all memory sequences.
     *
     * @param mem  memory manager pointer
     * @param data whether to clear data as well
     */
    void llama_memory_clear(LlamaMemoryManagerNative mem, boolean data);

    /**
     * Remove tokens from a sequence in the given range.
     *
     * @param mem    memory manager pointer
     * @param seq_id sequence ID
     * @param p0     start position (inclusive)
     * @param p1     end position (exclusive)
     * @return true if successful, false otherwise
     */
    boolean llama_memory_seq_rm(LlamaMemoryManagerNative mem, int seq_id, int p0, int p1);

    /**
     * Copy memory from one sequence to another.
     *
     * @param mem        memory manager pointer
     * @param seq_id_src source sequence ID
     * @param seq_id_dst destination sequence ID
     * @param p0         start position (inclusive)
     * @param p1         end position (exclusive)
     */
    void llama_memory_seq_cp(LlamaMemoryManagerNative mem, int seq_id_src, int seq_id_dst, int p0, int p1);

    /**
     * Keep only the specified sequence, removing all others.
     *
     * @param mem    memory manager pointer
     * @param seq_id sequence ID to keep
     */
    void llama_memory_seq_keep(LlamaMemoryManagerNative mem, int seq_id);

    /**
     * Add a delta to all positions in a sequence range.
     *
     * @param mem    memory manager pointer
     * @param seq_id sequence ID
     * @param p0     start position (inclusive)
     * @param p1     end position (exclusive)
     * @param delta  value to add to positions
     */
    void llama_memory_seq_add(LlamaMemoryManagerNative mem, int seq_id, int p0, int p1, int delta);

    /**
     * Divide positions in a sequence range by a divisor.
     *
     * @param mem    memory manager pointer
     * @param seq_id sequence ID
     * @param p0     start position (inclusive)
     * @param p1     end position (exclusive)
     * @param d      divisor
     */
    void llama_memory_seq_div(LlamaMemoryManagerNative mem, int seq_id, int p0, int p1, int d);

    /**
     * Get minimum position for a sequence.
     *
     * @param mem    memory manager pointer
     * @param seq_id sequence ID
     * @return minimum position, or -1 if sequence is empty
     */
    int llama_memory_seq_pos_min(LlamaMemoryManagerNative mem, int seq_id);

    /**
     * Get maximum position for a sequence.
     *
     * @param mem    memory manager pointer
     * @param seq_id sequence ID
     * @return maximum position, or -1 if sequence is empty
     */
    int llama_memory_seq_pos_max(LlamaMemoryManagerNative mem, int seq_id);

    /**
     * Check if memory can be shifted (context shifting is possible).
     *
     * @param mem memory manager pointer
     * @return true if shifting is possible, false otherwise
     */
    boolean llama_memory_can_shift(LlamaMemoryManagerNative mem);

    //
    // State Management Functions
    //

    /**
     * Get the size in bytes of the complete state data.
     *
     * @param ctx context pointer
     * @return size in bytes of the state data
     */
    long llama_state_get_size(LlamaContextNative ctx);

    /**
     * Copy state data to a buffer.
     *
     * @param ctx  context pointer
     * @param dst  destination buffer
     * @param size buffer size
     * @return number of bytes written, or negative on error
     */
    long llama_state_get_data(LlamaContextNative ctx, Pointer dst, long size);

    /**
     * Load state data from a buffer.
     *
     * @param ctx  context pointer
     * @param src  source buffer
     * @param size buffer size
     * @return number of bytes read, or negative on error
     */
    long llama_state_set_data(LlamaContextNative ctx, Pointer src, long size);

    /**
     * Load state from a file.
     *
     * @param ctx               context pointer
     * @param path_session      path to the session file
     * @param tokens_out        output buffer for tokens
     * @param n_token_capacity  capacity of token buffer
     * @param n_token_count_out pointer to store actual token count
     * @return true on success, false on failure
     */
    boolean llama_state_load_file(LlamaContextNative ctx, String path_session, Pointer tokens_out, long n_token_capacity, Pointer n_token_count_out);

    /**
     * Save state to a file.
     *
     * @param ctx           context pointer
     * @param path_session  path to the session file
     * @param tokens        token array to save
     * @param n_token_count number of tokens
     * @return true on success, false on failure
     */
    boolean llama_state_save_file(LlamaContextNative ctx, String path_session, Pointer tokens, long n_token_count);

    //
    // Sequence State Functions
    //

    /**
     * Get the size in bytes of sequence state data.
     *
     * @param ctx    context pointer
     * @param seq_id sequence ID
     * @return size in bytes of the sequence state
     */
    long llama_state_seq_get_size(LlamaContextNative ctx, int seq_id);

    /**
     * Copy sequence state data to a buffer.
     *
     * @param ctx    context pointer
     * @param dst    destination buffer
     * @param size   buffer size
     * @param seq_id sequence ID
     * @return number of bytes written, or negative on error
     */
    long llama_state_seq_get_data(LlamaContextNative ctx, Pointer dst, long size, int seq_id);

    /**
     * Load sequence state data from a buffer.
     *
     * @param ctx         context pointer
     * @param src         source buffer
     * @param size        buffer size
     * @param dest_seq_id destination sequence ID
     * @return number of bytes read, or negative on error
     */
    long llama_state_seq_set_data(LlamaContextNative ctx, Pointer src, long size, int dest_seq_id);

    /**
     * Save sequence state to a file.
     *
     * @param ctx           context pointer
     * @param filepath      path to the file
     * @param seq_id        sequence ID
     * @param tokens        token array to save
     * @param n_token_count number of tokens
     * @return number of bytes written, or negative on error
     */
    long llama_state_seq_save_file(LlamaContextNative ctx, String filepath, int seq_id, Pointer tokens, long n_token_count);

    /**
     * Load sequence state from a file.
     *
     * @param ctx               context pointer
     * @param filepath          path to the file
     * @param dest_seq_id       destination sequence ID
     * @param tokens_out        output buffer for tokens
     * @param n_token_capacity  capacity of token buffer
     * @param n_token_count_out pointer to store actual token count
     * @return number of bytes read, or negative on error
     */
    long llama_state_seq_load_file(LlamaContextNative ctx, String filepath, int dest_seq_id, Pointer tokens_out, long n_token_capacity, Pointer n_token_count_out);

    //
    // Extended Sequence State Functions
    //

    /**
     * Get the size in bytes of extended sequence state data.
     *
     * @param ctx    context pointer
     * @param seq_id sequence ID
     * @param flags  state flags (e.g., LLAMA_STATE_SEQ_FLAGS_SWA_ONLY)
     * @return size in bytes of the extended sequence state
     */
    long llama_state_seq_get_size_ext(LlamaContextNative ctx, int seq_id, int flags);

    /**
     * Copy extended sequence state data to a buffer.
     *
     * @param ctx    context pointer
     * @param dst    destination buffer
     * @param size   buffer size
     * @param seq_id sequence ID
     * @param flags  state flags
     * @return number of bytes written, or negative on error
     */
    long llama_state_seq_get_data_ext(LlamaContextNative ctx, Pointer dst, long size, int seq_id, int flags);

    /**
     * Load extended sequence state data from a buffer.
     *
     * @param ctx         context pointer
     * @param src         source buffer
     * @param size        buffer size
     * @param dest_seq_id destination sequence ID
     * @param flags       state flags
     * @return number of bytes read, or negative on error
     */
    long llama_state_seq_set_data_ext(LlamaContextNative ctx, Pointer src, long size, int dest_seq_id, int flags);

    // Vocabulary functions - already covered above

    //
    // Performance Functions
    //

    /**
     * Get performance statistics for a context.
     *
     * @param ctx context pointer
     * @return pointer to performance data structure
     */
    LlamaPerformanceContextDataNative llama_perf_context(LlamaContextNative ctx);

    /**
     * Print context performance statistics to stdout.
     *
     * @param ctx context pointer
     */
    void llama_perf_context_print(LlamaContextNative ctx);

    /**
     * Reset context performance counters.
     *
     * @param ctx context pointer
     */
    void llama_perf_context_reset(LlamaContextNative ctx);

    /**
     * Get performance statistics for a sampler chain.
     *
     * @param chain sampler chain pointer
     * @return pointer to performance data structure
     */
    LlamaPerformanceSamplerDataNative llama_perf_sampler(LlamaSamplerNative chain);

    /**
     * Print sampler performance statistics to stdout.
     *
     * @param smpl sampler pointer
     */
    void llama_perf_sampler_print(LlamaSamplerNative smpl);

    /**
     * Reset sampler performance counters.
     *
     * @param chain sampler chain pointer
     */
    void llama_perf_sampler_reset(LlamaSamplerNative chain);

    //
    // Optimization/Training Functions
    //

    /**
     * Filter function for optimization parameters (allows all tensors).
     *
     * @param tensor   tensor pointer
     * @param userdata user data
     * @return true to include tensor in optimization, false to exclude
     */
    boolean llama_opt_param_filter_all(Pointer tensor, Pointer userdata);

    /**
     * Initialize optimization context.
     *
     * @param lctx        llama context pointer
     * @param model       model pointer
     * @param lopt_params optimization parameters
     */
    void llama_opt_init(LlamaContextNative lctx, LlamaModelNative model, LlamaOptParamsNative lopt_params);

    /**
     * Run one optimization epoch.
     *
     * @param lctx           llama context pointer
     * @param dataset        training dataset
     * @param result_train   training results output
     * @param result_eval    evaluation results output
     * @param idata_split    data split index
     * @param callback_train training callback function
     * @param callback_eval  evaluation callback function
     */
    void llama_opt_epoch(LlamaContextNative lctx, Pointer dataset, Pointer result_train, Pointer result_eval, long idata_split, Pointer callback_train, Pointer callback_eval);

    //
    // Adapter Functions
    //

    /**
     * Load a LoRA adapter from file.
     * <p>
     * Creates a LoRA adapter that can be applied to contexts for fine-tuned behavior.
     * The adapter will be automatically freed when the associated model is deleted.
     *
     * @param model     model pointer
     * @param path_lora path to the LoRA adapter file
     * @return pointer to the loaded LoRA adapter, or null on failure
     */
    LlamaAdapterNative llama_adapter_lora_init(LlamaModelNative model, String path_lora);

    /**
     * Get LoRA adapter metadata value as string by key name.
     * <p>
     * Functions to access the adapter's GGUF metadata scalar values.
     * The output string is always null-terminated and cleared on failure.
     * When retrieving a string, an extra byte must be allocated for the null terminator.
     * GGUF array values are not supported by these functions.
     *
     * @param adapter  LoRA adapter pointer
     * @param key      metadata key name
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the string on success, or -1 on failure
     */
    int llama_adapter_meta_val_str(LlamaAdapterNative adapter, String key, byte[] buf, long buf_size);

    /**
     * Get the number of metadata key/value pairs in the LoRA adapter.
     *
     * @param adapter LoRA adapter pointer
     * @return number of metadata pairs
     */
    int llama_adapter_meta_count(LlamaAdapterNative adapter);

    /**
     * Get LoRA adapter metadata key name by index.
     *
     * @param adapter  LoRA adapter pointer
     * @param i        metadata index
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the key name on success, or -1 on failure
     */
    int llama_adapter_meta_key_by_index(LlamaAdapterNative adapter, int i, byte[] buf, long buf_size);

    /**
     * Get LoRA adapter metadata value as string by index.
     *
     * @param adapter  LoRA adapter pointer
     * @param i        metadata index
     * @param buf      output buffer
     * @param buf_size buffer size
     * @return length of the value string on success, or -1 on failure
     */
    int llama_adapter_meta_val_str_by_index(LlamaAdapterNative adapter, int i, byte[] buf, long buf_size);

    /**
     * Manually freeSampler a LoRA adapter.
     * <p>
     * Note: loaded adapters will be automatically freed when the associated model is deleted.
     *
     * @param adapter LoRA adapter pointer to freeSampler
     */
    void llama_adapter_lora_free(LlamaAdapterNative adapter);

    /**
     * Get the number of invocation tokens if the current LoRA is an ALoRA.
     *
     * @param adapter LoRA adapter pointer
     * @return number of invocation tokens
     */
    long llama_adapter_get_alora_n_invocation_tokens(LlamaAdapterNative adapter);

    /**
     * Get the invocation tokens if the current LoRA is an ALoRA.
     *
     * @param adapter LoRA adapter pointer
     * @return pointer to invocation tokens array
     */
    Pointer llama_adapter_get_alora_invocation_tokens(LlamaAdapterNative adapter);

    /**
     * Add a loaded LoRA adapter to given context.
     * <p>
     * This will not modify the model's weights directly but applies the adapter
     * during inference. The following functions operate on a llama_context.
     *
     * @param ctx     context pointer
     * @param adapter LoRA adapter pointer
     * @param scale   scaling factor for the adapter
     * @return 0 on success, negative on error
     */
    int llama_set_adapter_lora(LlamaContextNative ctx, LlamaAdapterNative adapter, float scale);

    /**
     * Remove a specific LoRA adapter from given context.
     *
     * @param ctx     context pointer
     * @param adapter LoRA adapter pointer to remove
     * @return 0 on success, -1 if the adapter is not present in the context
     */
    int llama_rm_adapter_lora(LlamaContextNative ctx, LlamaAdapterNative adapter);

    /**
     * Remove all LoRA adapters from given context.
     *
     * @param ctx context pointer
     */
    void llama_clear_adapter_lora(LlamaContextNative ctx);

    /**
     * Apply a loaded control vector to a llama_context, or if data is NULL, clear
     * the currently loaded vector.
     * <p>
     * n_embd should be the size of a single layer's control, and data should point
     * to an n_embd x n_layers buffer starting from layer 1.
     * il_start and il_end are the layer range the vector should apply to (both inclusive).
     * See llama_control_vector_load in common to load a control vector.
     *
     * @param ctx      context pointer
     * @param data     pointer to control vector data (float array), or null to clear
     * @param len      length of the data buffer
     * @param n_embd   embedding dimension (size of single layer control)
     * @param il_start start layer index (inclusive)
     * @param il_end   end layer index (inclusive)
     * @return 0 on success, negative on error
     */
    int llama_apply_adapter_cvec(LlamaContextNative ctx, Pointer data, long len, int n_embd, int il_start, int il_end);

    //
    // Model Split Functions
    //

    /**
     * Generate split file path for model sharding.
     *
     * @param split_path  output buffer for the split path
     * @param maxlen      maximum length of output buffer
     * @param path_prefix path prefix for split files
     * @param split_no    split number (0-based)
     * @param split_count total number of splits
     * @return length of generated path, or negative on error
     */
    int llama_split_path(byte[] split_path, long maxlen, String path_prefix, int split_no, int split_count);

    /**
     * Generate split file prefix from split path.
     *
     * @param split_prefix output buffer for the split prefix
     * @param maxlen       maximum length of output buffer
     * @param split_path   input split file path
     * @param split_no     split number (0-based)
     * @param split_count  total number of splits
     * @return length of generated prefix, or negative on error
     */
    int llama_split_prefix(byte[] split_prefix, long maxlen, String split_path, int split_no, int split_count);
}

