package com.quasarbyte.llama.cpp.jna.examples.simplechat;

import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibraryFactory;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaConstants;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibraryFactory;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static com.quasarbyte.llama.cpp.jna.binding.ggml.backend.GgmlBackendConstants.GGML_BACKEND_PATH_ENV_VAR_NAME;

/**
 * SimpleChat - Java implementation that closely follows the original C++ simple-chat.cpp logic.
 * <p>
 * This implementation demonstrates direct usage of the llama.cpp library through JNA bindings,
 * mirroring the approach used in the C++ example rather than using the high-level binding layer.
 * <p>
 * Key characteristics matching C++ version:
 * - Direct library function calls instead of binding abstractions
 * - Single generate function containing core logic
 * - Minimal error handling (abort on errors)
 * - Simple chat template handling
  * - Basic memory management
 * <p>
 * JVM-specific notes:
 * <ul>
 *   <li>{@code ggml_backend_load_all()} only scans the executable and current directories;
 *       when backend binaries live elsewhere load them explicitly (for example via
 *       {@code ggml_backend_load_all_from_path(System.getenv(GGML_BACKEND_PATH_ENV_VAR_NAME))}).</li>
 *   <li>After the GGML plugins are present, call {@code llama_backend_init()} before creating
 *       models or contexts and mirror it with {@code llama_backend_free()} during shutdown so
 *       device resources are released.</li>
 * </ul>
 */
public class SimpleChat {

    private static final String GREEN = "\033[32m";
    private static final String YELLOW = "\033[33m";
    private static final String RESET = "\033[0m";
    public static final String APP_STDIN_ENCODING = "app.stdin.encoding";

    private static void printUsage(String programName) {
        System.out.println();
        System.out.println("example usage:");
        System.out.println();
        System.out.printf("    %s -m model.gguf [-c context_size] [-ngl n_gpu_layers]%n", programName);
        System.out.println();
    }

    public static void main(String[] args) {
        String modelPath = "";
        int ngl = 99;
        int nCtx = 2048;

        // Parse command line arguments - exactly like C++ version
        for (int i = 0; i < args.length; i++) {
            try {
                if ("-m".equals(args[i])) {
                    if (i + 1 < args.length) {
                        modelPath = args[++i];
                    } else {
                        printUsage("SimpleChat");
                        System.exit(1);
                    }
                } else if ("-c".equals(args[i])) {
                    if (i + 1 < args.length) {
                        nCtx = Integer.parseInt(args[++i]);
                    } else {
                        printUsage("SimpleChat");
                        System.exit(1);
                    }
                } else if ("-ngl".equals(args[i])) {
                    if (i + 1 < args.length) {
                        ngl = Integer.parseInt(args[++i]);
                    } else {
                        printUsage("SimpleChat");
                        System.exit(1);
                    }
                } else {
                    printUsage("SimpleChat");
                    System.exit(1);
                }
            } catch (NumberFormatException e) {
                System.err.println("error: " + e.getMessage());
                printUsage("SimpleChat");
                System.exit(1);
            }
        }

        if (modelPath.isEmpty()) {
            printUsage("SimpleChat");
            System.exit(1);
        }

        // Initialize libraries - mirroring C++ approach
        GgmlLibrary ggmlLibrary = new GgmlLibraryFactory().getInstance();

        ggmlLibrary.ggml_backend_load_all_from_path(System.getenv(GGML_BACKEND_PATH_ENV_VAR_NAME));

        LlamaLibrary llamaLibrary = new LlamaLibraryFactory().getInstance();

        llamaLibrary.llama_backend_init();

        // Only print errors - exactly like C++ version
        llamaLibrary.llama_log_set(new LlamaLibrary.LlamaLogCallback() {
            @Override
            public void invoke(int level, String text, Pointer userData) {
                if (level >= LlamaLibrary.GGML_LOG_LEVEL_ERROR) {
                    System.err.print(text);
                }
            }
        }, null);

        // Load dynamic backends
        ggmlLibrary.ggml_backend_load_all();

        // Initialize the model - direct library calls like C++
        LlamaModelParamsNative modelParams = llamaLibrary.llama_model_default_params();
        modelParams.n_gpu_layers = ngl;

        LlamaModelNative model = llamaLibrary.llama_model_load_from_file(modelPath, modelParams);
        if (model == null) {
            System.err.println("error: unable to load model");
            System.exit(1);
        }

        LlamaVocabularyNative vocab = llamaLibrary.llama_model_get_vocab(model);

        // Initialize the context - direct library calls like C++
        LlamaContextParamsNative ctxParams = llamaLibrary.llama_context_default_params();
        ctxParams.n_ctx = nCtx;
        ctxParams.n_batch = nCtx;

        LlamaContextNative ctx = llamaLibrary.llama_init_from_model(model, ctxParams);
        if (ctx == null) {
            System.err.println("error: failed to create the llama_context");
            System.exit(1);
        }

        // Initialize the sampler - exactly like C++ version
        LlamaSamplerNative smpl = llamaLibrary.llama_sampler_chain_init(llamaLibrary.llama_sampler_chain_default_params());
        llamaLibrary.llama_sampler_chain_add(smpl, llamaLibrary.llama_sampler_init_min_p(0.05f, 1));
        llamaLibrary.llama_sampler_chain_add(smpl, llamaLibrary.llama_sampler_init_temp(0.8f));
        llamaLibrary.llama_sampler_chain_add(smpl, llamaLibrary.llama_sampler_init_dist(LlamaConstants.LLAMA_DEFAULT_SEED));

        // Helper function to evaluate a prompt and generate a response - like C++ lambda
        Generate generate = new Generate(llamaLibrary, ctx, vocab, smpl);

        // Main chat loop - exactly like C++ version
        List<LlamaChatMessageNative> messages = new ArrayList<>();

        int prevLen = 0;

        String appStdinEncoding = System.getProperty(APP_STDIN_ENCODING);

        final Charset charset;

        if (appStdinEncoding != null) {
            charset = Charset.forName(appStdinEncoding);
        } else {
            charset = StandardCharsets.UTF_8;
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, charset));

        try (Memory formatted = new Memory(llamaLibrary.llama_n_ctx(ctx))) {

            while (true) {
                // Get user input
                System.out.print(GREEN + "> " + RESET);
                String user = reader.readLine();

                if (user == null || user.trim().isEmpty()) {
                    break;
                }

                String tmpl = llamaLibrary.llama_model_chat_template(model, null);

                // Add the user input to the message list and format it
                LlamaChatMessageNative userMessage = new LlamaChatMessageNative("user", user);
                messages.add(userMessage);

                LlamaChatMessageNative[] messagesArray = messagesToArray(messages);

                byte[] formattedBuffer = new byte[(int) formatted.size()];
                int newLen = llamaLibrary.llama_chat_apply_template(tmpl, messagesArray, messagesArray.length, true, formattedBuffer, formattedBuffer.length);

                if (newLen > formattedBuffer.length) {
                    formattedBuffer = new byte[newLen];
                    newLen = llamaLibrary.llama_chat_apply_template(tmpl, messagesArray, messagesArray.length, true, formattedBuffer, formattedBuffer.length);
                }

                if (newLen < 0) {
                    System.err.println("failed to apply the chat template");
                    System.exit(1);
                }

                // Remove previous messages to obtain the prompt to generate the response
                String prompt = new String(formattedBuffer, prevLen, newLen - prevLen, StandardCharsets.UTF_8);

                // Generate a response
                System.out.print(YELLOW);
                String response = generate.call(prompt);
                System.out.print("\n" + RESET);

                // Add the response to the messages
                LlamaChatMessageNative assistantMessage = new LlamaChatMessageNative("assistant", response);
                messages.add(assistantMessage);

                messagesArray = messagesToArray(messages);
                prevLen = llamaLibrary.llama_chat_apply_template(tmpl, messagesArray, messagesArray.length, false, null, 0);
                if (prevLen < 0) {
                    System.err.println("failed to apply the chat template");
                    System.exit(1);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading input: " + e.getMessage());
            System.exit(1);
        }

        // Free resources - exactly like C++ version
        llamaLibrary.llama_sampler_free(smpl);
        llamaLibrary.llama_free(ctx);
        llamaLibrary.llama_model_free(model);
        llamaLibrary.llama_backend_free();
    }

    /**
     * Generate class - mimics the C++ lambda function for text generation
     */
    private static class Generate {
        private final LlamaLibrary llamaLibrary;
        private final LlamaContextNative ctx;
        private final LlamaVocabularyNative vocab;
        private final LlamaSamplerNative smpl;

        public Generate(LlamaLibrary llamaLibrary, LlamaContextNative ctx, LlamaVocabularyNative vocab, LlamaSamplerNative smpl) {
            this.llamaLibrary = llamaLibrary;
            this.ctx = ctx;
            this.vocab = vocab;
            this.smpl = smpl;
        }

        public String call(String prompt) {
            StringBuilder response = new StringBuilder();

            // Resolve context & memory state
            LlamaMemoryManagerNative memory = llamaLibrary.llama_get_memory(ctx);
            boolean isFirst = llamaLibrary.llama_memory_seq_pos_max(memory, 0) == -1;

            // --- Tokenize the prompt (UTF-8 bytes + explicit byte length) ---
            // 1) probe for required token count
            byte[] utf8 = prompt.getBytes(StandardCharsets.UTF_8);

            int probe = llamaLibrary.llama_tokenize(
                    vocab,
                    utf8,                 // pass bytes, NOT Java String
                    utf8.length,          // length in bytes
                    Pointer.NULL,         // no output buffer yet
                    0,                    // n_tokens_max = 0
                    isFirst,              // add BOS if model demands and this is first seq
                    true                  // allow special tokens (mirrors your original)
            );

            if (probe == Integer.MIN_VALUE) {
                throw new LlamaFunctionCallIntResultException(
                        probe, "Tokenization result size exceeds int32_t limit");
            }

            // Required tokens: positive means fits (often 0 for empty); negative => need -probe
            int required = (probe >= 0) ? probe : -probe;
            if (required == 0) {
                // Empty prompt – nothing to decode; return empty response
                return "";
            }

            // Allocate native memory for prompt tokens (int32_t = 4 bytes)
            try (Memory promptTokensMemory = new Memory((long) required * 4L)) {

                int t2 = llamaLibrary.llama_tokenize(
                        vocab,
                        utf8,
                        utf8.length,
                        promptTokensMemory,  // output buffer
                        required,
                        isFirst,
                        true
                );

                if (t2 == Integer.MIN_VALUE) {
                    throw new LlamaFunctionCallIntResultException(
                            t2, "Overflow: tokenization result size exceeds int32_t limit");
                }
                if (t2 < 0) {
                    int need = -t2;
                    throw new LlamaFunctionCallIntResultException(
                            t2, "Prompt token buffer too small, need at least " + need);
                }

                int nPromptTokens = t2; // use the actual count returned

                // Prepare an initial batch from the prompt tokens.
                // NOTE: llama_batch_get_one is assumed to reference the provided memory,
                // so 'promptTokensMemory' must remain alive while the batch is used.
                LlamaBatchNative batch = llamaLibrary.llama_batch_get_one(promptTokensMemory, nPromptTokens);

                // Persistent 1-token native buffer reused for streaming new tokens.
                // IMPORTANT: keep this Memory alive across loop iterations (do NOT close inside the loop),
                // because batch will refer to it.
                Memory oneTokenMem = new Memory(4L);

                int generatedTokens = 0;
                final int MAX_GENERATION_LENGTH = 2048;    // guardrail to avoid infinite loops

                while (generatedTokens < MAX_GENERATION_LENGTH) {
                    // Ensure we have context space for this batch
                    int nCtx = (int) llamaLibrary.llama_n_ctx(ctx);
                    int nCtxUsed = llamaLibrary.llama_memory_seq_pos_max(memory, 0) + 1;
                    if (nCtxUsed + batch.n_tokens > nCtx) {
                        // Prefer throwing to System.exit in libraries
                        throw new IllegalStateException("Context size exceeded");
                    }

                    int ret = llamaLibrary.llama_decode(ctx, batch);
                    if (ret != 0) {
                        throw new IllegalStateException("Failed to decode, ret = " + ret);
                    }

                    // Sample next token
                    int newTokenId = llamaLibrary.llama_sampler_sample(smpl, ctx, -1);

                    // EOS/EOT check (direct token ID comparison)
                    int modelEosToken = llamaLibrary.llama_vocab_eos(vocab);
                    int modelEotToken = llamaLibrary.llama_vocab_eot(vocab);
                    if (newTokenId == modelEosToken || newTokenId == modelEotToken) {
                        break;
                    }

                    // Convert token to text and append to response/console
                    {
                        byte[] buf = new byte[256];
                        int n = llamaLibrary.llama_token_to_piece(vocab, newTokenId, buf, buf.length, 0, true);
                        if (n < 0) {
                            throw new IllegalStateException("Failed to convert token to piece");
                        }
                        String piece = new String(buf, 0, n, StandardCharsets.UTF_8);
                        System.out.print(piece);
                        System.out.flush();
                        response.append(piece);
                    }

                    generatedTokens++;

                    // Prepare the next batch from the sampled token (reuse persistent oneTokenMem)
                    oneTokenMem.setInt(0, newTokenId);
                    batch = llamaLibrary.llama_batch_get_one(oneTokenMem, 1);
                }
            }

            return response.toString();
        }
    }

    private static LlamaChatMessageNative[] messagesToArray(List<LlamaChatMessageNative> messages) {
        if (messages.isEmpty()) {
            return new LlamaChatMessageNative[0];
        }

        // Create contiguous array using JNA's array allocation
        LlamaChatMessageNative firstMessage = new LlamaChatMessageNative();
        LlamaChatMessageNative[] array = (LlamaChatMessageNative[]) firstMessage.toArray(messages.size());

        for (int i = 0; i < messages.size(); i++) {
            LlamaChatMessageNative source = messages.get(i);
            array[i].setRole(source.getRole());
            array[i].setContent(source.getContent());
        }

        return array;
    }
}

