package com.quasarbyte.llama.cpp.jna.examples.simplechat;

import com.quasarbyte.llama.cpp.jna.binding.llama.chat.LlamaChatBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.processing.LlamaProcessingBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary.LlamaVocabularyBinding;
import com.quasarbyte.llama.cpp.jna.binding.logging.CustomLlamaLoggerBinding;
import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibraryFactory;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibraryFactory;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaConstants;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader.GgmlBackendLoader;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader.GgmlBackendLoaderFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.backend.LlamaBackendBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.backend.LlamaBackendBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.batch.LlamaBatchBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.batch.LlamaBatchBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.chat.LlamaChatBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.context.LlamaContextBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.context.LlamaContextBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.memory.LlamaMemoryBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.memory.LlamaMemoryBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.model.LlamaModelBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.model.LlamaModelBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.processing.LlamaProcessingBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.sampler.LlamaSamplerBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.sampler.LlamaSamplerBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.token.LlamaTokenBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.token.LlamaTokenBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary.LlamaVocabularyBindingFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class SimpleChatLayered {
    private static final Logger logger = LoggerFactory.getLogger(SimpleChatLayered.class);
    private static final int DEFAULT_N_CTX = 32768;
    private static final int DEFAULT_N_GPU_LAYERS = 99;
    private static final int TOKEN_TO_PIECE_MAX_LENGTH = 128;
    private static final int DEFAULT_BUFFER_SIZE = 1024 * 64;
    private static final int MAX_GENERATION_TOKENS = 10000;
    public static final String APP_STDIN_ENCODING = "app.stdin.encoding";

    private LlamaLibrary llamaLibrary;
    private LlamaModelBinding modelLoaderService;
    private LlamaVocabularyBinding vocabularyService;

    // ANSI color codes for terminal output
    private static final String GREEN = "\033[32m";
    private static final String YELLOW = "\033[33m";
    private static final String RESET = "\033[0m";

    private static void printUsage() {
        System.out.println();
        System.out.println("example usage:");
        System.out.println();
        System.out.printf("  %s -m model.gguf [-c context_size] [-ngl n_gpu_layers]%n", SimpleChatLayered.class.getName());
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -m            Model file path (required)");
        System.out.println("  -c            Context size (default: 2048)");
        System.out.println("  -ngl          Number of GPU layers (default: 99)");
        System.out.println();
    }

    public static void main(String[] args) {
        SimpleChatLayered chat = new SimpleChatLayered();
        chat.run(args);
    }

    private void run(String[] args) {
        GgmlLibrary ggmlLibrary = new GgmlLibraryFactory().getInstance();
        GgmlBackendLoader backendLoaderService = new GgmlBackendLoaderFactory().create(ggmlLibrary);

        this.llamaLibrary = new LlamaLibraryFactory().getInstance();

        // Setup custom logging to capture llama.cpp log messages
        CustomLlamaLoggerBinding.setupCustomLogging(llamaLibrary);

        LlamaBackendBinding backendService = new LlamaBackendBindingFactory().create(llamaLibrary);

        this.modelLoaderService = new LlamaModelBindingFactory().create(llamaLibrary);

        this.vocabularyService = new LlamaVocabularyBindingFactory().create(llamaLibrary);

        LlamaTokenBinding tokenService = new LlamaTokenBindingFactory().create(llamaLibrary);
        LlamaBatchBinding batchService = new LlamaBatchBindingFactory().create(llamaLibrary);
        LlamaContextBinding contextService = new LlamaContextBindingFactory().create(llamaLibrary);
        LlamaSamplerBinding samplerService = new LlamaSamplerBindingFactory().create(llamaLibrary);
        LlamaProcessingBinding processingService = new LlamaProcessingBindingFactory().create(llamaLibrary);
        LlamaChatBinding chatService = new LlamaChatBindingFactory().create(llamaLibrary);
        LlamaMemoryBinding memoryService = new LlamaMemoryBindingFactory().create(llamaLibrary);

        CommandLineArgs cmdArgs = parseCommandLineArgs(args);
        if (cmdArgs.modelPath.isEmpty()) {
            printUsage();
            System.exit(1);
        }

        try {
            logger.info("Starting llama.cpp chat initialization...");

            backendLoaderService.loadBackend();
            backendService.backendInit();

            LlamaModel model = loadModel(modelLoaderService, Paths.get(cmdArgs.modelPath), cmdArgs.nGpuLayers);

            LlamaVocabulary vocabulary = this.vocabularyService.getVocabulary(model);

            // Log automatically detected special tokens
            logger.info("Auto-detected special tokens:\n{}", getSpecialTokensSummary(vocabulary));

            LlamaContext context = createContext(contextService, model, cmdArgs.nCtx, cmdArgs.nCtx);
            LlamaSampler sampler = createChatSampler(samplerService);

            logger.info("Chat initialized successfully. Type your messages (empty line to exit):");
            runChatLoop(tokenService, batchService, contextService, samplerService, processingService,
                    chatService, memoryService, model, vocabulary, context, sampler);

            cleanup(samplerService, contextService, modelLoaderService, backendService,
                    sampler, context, model);

        } catch (Exception e) {
            logger.error("Error: {}", e.getMessage(), e);
            System.err.println(e.getMessage());
            e.getStackTrace();
            System.exit(1);
        }
    }

    private CommandLineArgs parseCommandLineArgs(String[] args) {
        CommandLineArgs cmdArgs = new CommandLineArgs();

        int i = 0;
        while (i < args.length) {
            try {
                if ("-m".equals(args[i])) {
                    if (i + 1 < args.length) {
                        cmdArgs.modelPath = args[++i];
                    } else {
                        printUsage();
                        System.exit(1);
                    }
                } else if ("-c".equals(args[i])) {
                    if (i + 1 < args.length) {
                        cmdArgs.nCtx = Integer.parseInt(args[++i]);
                    } else {
                        printUsage();
                        System.exit(1);
                    }
                } else if ("-ngl".equals(args[i])) {
                    if (i + 1 < args.length) {
                        cmdArgs.nGpuLayers = Integer.parseInt(args[++i]);
                    } else {
                        printUsage();
                        System.exit(1);
                    }
                } else {
                    printUsage();
                    System.exit(1);
                }
            } catch (NumberFormatException e) {
                logger.error("Invalid number format for argument: {}", args[i]);
                printUsage();
                System.exit(1);
            }
            i++;
        }

        return cmdArgs;
    }

    private LlamaModel loadModel(LlamaModelBinding modelLoaderService,
                                 Path modelPath,
                                 int nGpuLayers) {
        logger.info("Loading model from: {}", modelPath);
        try {
            LlamaModel model = modelLoaderService.loadFromFile(modelPath, nGpuLayers);
            logger.info("Model loaded successfully");
            return model;
        } catch (Exception e) {
            logger.error("Failed to load model: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to load model: %s", e.getMessage()), e);
        }
    }

    private LlamaContext createContext(LlamaContextBinding contextService,
                                       LlamaModel model,
                                       int nCtx,
                                       int nBatch) {
        logger.debug("Creating context with n_ctx={}, n_batch={}", nCtx, nBatch);
        try {
            LlamaContext context = contextService.create(model, nCtx, nBatch, false);
            logger.debug("Context created successfully");
            return context;
        } catch (Exception e) {
            logger.error("Failed to create context: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to create context: %s", e.getMessage()), e);
        }
    }

    private LlamaSampler createChatSampler(LlamaSamplerBinding samplerService) {
        logger.debug("Creating chat sampler...");
        try {
            // Create a more sophisticated sampler for chat (similar to C++ example)
            LlamaSampler sampler = samplerService.createChainSampler(false);

            // Add min_p sampling
            LlamaSampler minP = samplerService.createMinPSampler(0.05f, 1);
            samplerService.addSamplerToChain(minP, sampler);

            // Add temperature sampling
            LlamaSampler temp = samplerService.createTemperatureSampler(0.8f);
            samplerService.addSamplerToChain(temp, sampler);

            // Add distribution sampling
            LlamaSampler dist = samplerService.createDistributionSampler(LlamaConstants.LLAMA_DEFAULT_SEED);
            samplerService.addSamplerToChain(dist, sampler);

            logger.debug("Chat sampler created successfully");
            return sampler;
        } catch (Exception e) {
            logger.error("Failed to create chat sampler: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to create chat sampler: %s", e.getMessage()), e);
        }
    }

    private void runChatLoop(LlamaTokenBinding tokenService,
                             LlamaBatchBinding batchService,
                             LlamaContextBinding contextService,
                             LlamaSamplerBinding samplerService,
                             LlamaProcessingBinding processingService,
                             LlamaChatBinding chatService,
                             LlamaMemoryBinding memoryService,
                             LlamaModel model,
                             LlamaVocabulary vocabulary,
                             LlamaContext context,
                             LlamaSampler sampler) {

        List<LlamaChatMessage> messages = new ArrayList<>();

        String appStdinEncoding = System.getProperty(APP_STDIN_ENCODING);

        final Charset charset;

        if (appStdinEncoding != null) {
            charset = Charset.forName(appStdinEncoding);
        } else {
            charset = StandardCharsets.UTF_8;
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, charset));

        int prevLen = 0;

        // Get the model's chat template - use native approach with fallback
        String chatTemplate = null;
        try {
            chatTemplate = chatService.getChatTemplate(model, null).orElse(null);
            if (chatTemplate == null || chatTemplate.trim().isEmpty()) {
                logger.warn("No chat template found, will use fallback format");
            } else {
                logger.info("Using model's chat template");
            }
        } catch (Exception e) {
            logger.warn("Failed to get chat template, using fallback: {}", e.getMessage());
        }

        try {
            while (true) {
                // Get user input
                System.out.print(GREEN + "> " + RESET);
                String userInput = reader.readLine();

                if (userInput == null || userInput.trim().isEmpty()) {
                    break;
                }

                // Add user message to conversation
                messages.add(chatService.createUserMessage(userInput.trim()));

                // Apply chat template with fallback on error
                String formatted;
                if (chatTemplate != null) {
                    try {
                        formatted = chatService.applyChatTemplate(chatTemplate, messages, true, DEFAULT_BUFFER_SIZE);
                        if (formatted == null || formatted.isEmpty()) {
                            logger.warn("Chat template returned empty result, using fallback");
                            formatted = applyChatTemplateFallback(messages, true);
                        }
                    } catch (Exception e) {
                        logger.warn("Chat template failed, using fallback: {}", e.getMessage());
                        formatted = applyChatTemplateFallback(messages, true);
                    }
                } else {
                    formatted = applyChatTemplateFallback(messages, true);
                }

                // Extract the prompt for generation (remove previous messages)
                String prompt = formatted.substring(prevLen);
                logger.debug("Generated prompt: {}", prompt);

                // Generate response
                System.out.print(YELLOW);
                String response = generateResponse(tokenService, batchService, contextService, samplerService,
                        processingService, memoryService, model, vocabulary, context, sampler, prompt);
                System.out.print("\n" + RESET);

                // Add assistant response to conversation
                messages.add(chatService.createAssistantMessage(response));

                // Update previous length for next iteration
                if (chatTemplate != null) {
                    try {
                        String updatedFormatted = chatService.applyChatTemplate(chatTemplate, messages, false, DEFAULT_BUFFER_SIZE);
                        if (updatedFormatted != null) {
                            prevLen = updatedFormatted.length();
                        } else {
                            String fallbackFormatted = applyChatTemplateFallback(messages, false);
                            prevLen = fallbackFormatted.length();
                        }
                    } catch (Exception e) {
                        String fallbackFormatted = applyChatTemplateFallback(messages, false);
                        prevLen = fallbackFormatted.length();
                    }
                } else {
                    String updatedFormatted = applyChatTemplateFallback(messages, false);
                    prevLen = updatedFormatted.length();
                }
            }
        } catch (IOException e) {
            logger.error("Error reading user input: {}", e.getMessage(), e);
            throw new RuntimeException(String.format("Error reading user input: %s", e.getMessage()), e);
        }
    }

    private String applyChatTemplateFallback(List<LlamaChatMessage> messages, boolean addAssistant) {
        StringBuilder formatted = new StringBuilder();

        for (LlamaChatMessage message : messages) {
            if ("user".equals(message.getRole())) {
                formatted.append("User: ").append(message.getContent()).append("\n");
            } else if ("assistant".equals(message.getRole())) {
                formatted.append("Assistant: ").append(message.getContent()).append("\n");
            } else if ("system".equals(message.getRole())) {
                formatted.append("System: ").append(message.getContent()).append("\n");
            }
        }

        // Add the prompt for the next assistant response
        if (addAssistant) {
            formatted.append("Assistant: ");
        }

        return formatted.toString();
    }

    private String generateResponse(LlamaTokenBinding tokenService,
                                    LlamaBatchBinding batchService,
                                    LlamaContextBinding contextService,
                                    LlamaSamplerBinding samplerService,
                                    LlamaProcessingBinding processingService,
                                    LlamaMemoryBinding memoryService,
                                    LlamaModel model,
                                    LlamaVocabulary vocabulary,
                                    LlamaContext context,
                                    LlamaSampler sampler,
                                    String prompt) {

        StringBuilder response = new StringBuilder();

        try {
            // Check if this is the first message using memory binding
            LlamaMemoryManager memory = memoryService.getMemory(context);
            boolean isFirst = memoryService.sequencePositionMax(memory, 0) == -1;

            // Tokenize the prompt with automatic BOS detection
            boolean addBos = isFirst && vocabularyService.getAddBos(vocabulary);
            int[] promptTokens = tokenService.tokenize(model, vocabulary, prompt, addBos, true);
            logger.debug("Prompt tokenized into {} tokens", promptTokens.length);

            // Prepare batch for the prompt
            LlamaBatch batch = batchService.createBatch(promptTokens);

            int tokenCount = 0;
            while (true) {
                tokenCount++;

                // Check max token limit to prevent excessive generation
                if (tokenCount > MAX_GENERATION_TOKENS) {
                    logger.warn("Max token limit ({}) reached, stopping generation", MAX_GENERATION_TOKENS);
                    break;
                }

                // Check context space before each decode
                long nCtx = contextService.getContextSize(context);
                int nCtxUsed = memoryService.sequencePositionMax(memory, 0) + 1;

                if (nCtxUsed + batch.getTokenCount() > nCtx) {
                    System.out.print(RESET);
                    System.err.println("Context size exceeded");
                    System.exit(0);
                }

                // Decode the current batch
                int result = processingService.decodeBatch(context, batch);
                if (result != 0) {
                    logger.error("Decode failed with result: {} at iteration {}", result, tokenCount);
                    throw new RuntimeException(String.format("Failed to decode, result = %d", result));
                }

                // Sample the next token
                int newTokenId = samplerService.sampleToken(sampler, context, -1);

                // Accept the sampled token to update sampler state (critical for proper sampling)
                samplerService.acceptToken(sampler, newTokenId);

                // Convert token to see what it represents
                String debugPiece = tokenService.tokenToPiece(model, vocabulary, newTokenId, true, TOKEN_TO_PIECE_MAX_LENGTH);

                // Check if it's end of generation
                boolean isEos = tokenService.isEndOfGeneration(vocabulary, newTokenId);
                logger.debug("EOS check for token {}: {}", newTokenId, isEos);

                // Smart EOS detection: Use automatic detection
                boolean isLegitimateEos = isLegitimateEndOfGeneration(vocabulary, newTokenId);

                if (isLegitimateEos) {
                    logger.info("End of generation: token {} ('{}')",
                               newTokenId, debugPiece != null ? debugPiece.replace("\n", "\\n").replace("\r", "\\r") : "null");
                    break;
                } else if (isEos) {
                    // Not a legitimate EOS according to our binding - continue generation
                    logger.debug("Ignoring non-legitimate EOS token {} ('{}') - continuing generation",
                               newTokenId, debugPiece != null ? debugPiece.replace("\n", "\\n").replace("\r", "\\r") : "null");
                }

                // Convert token to text and print it
                String piece = tokenService.tokenToPiece(model, vocabulary, newTokenId, true, TOKEN_TO_PIECE_MAX_LENGTH);
                if (piece != null && !piece.isEmpty()) {
                    System.out.print(piece);
                    System.out.flush();
                    response.append(piece);
                }

                // Prepare next batch with the sampled token
                batchService.freeBatch(batch);
                batch = batchService.createBatch(new int[]{newTokenId});
            }

            logger.info("Generation completed after {} iterations, response length: {}", tokenCount, response.length());

            // Clean up the final batch
            batchService.freeBatch(batch);

        } catch (Exception e) {
            logger.error("Error during response generation: {} (Type: {})",
                         e.getMessage() != null ? e.getMessage() : "No message",
                         e.getClass().getSimpleName(), e);
            String errorMsg = e.getMessage() != null ? e.getMessage() : "Unknown error (" + e.getClass().getSimpleName() + ")";
            return "Error generating response: " + errorMsg;
        }

        return response.toString().trim();
    }

    private void cleanup(LlamaSamplerBinding samplerService,
                         LlamaContextBinding contextService,
                         LlamaModelBinding modelLoaderService,
                         LlamaBackendBinding backendService,
                         LlamaSampler sampler,
                         LlamaContext context,
                         LlamaModel model) {
        logger.debug("Starting cleanup...");

        try {
            safelyFree("Sampler", () -> samplerService.freeSampler(sampler));
            safelyFree("Context", () -> contextService.freeContext(context));
            safelyFree("Model", () -> modelLoaderService.freeModel(model));
            safelyFree("Backend", backendService::freeBackend);
        } catch (Exception e) {
            logger.error("Failed to cleanup: {}", e.getMessage());
            throw new RuntimeException("Failed to cleanup: " + e.getMessage());
        }

        logger.debug("Cleanup completed successfully");
    }

    private void safelyFree(String resourceName,
                            Runnable freeOperation) {
        try {
            freeOperation.run();
            logger.debug("{} freed", resourceName);
        } catch (Exception e) {
            logger.error("Failed to freeSampler {}: {}", resourceName.toLowerCase(), e.getMessage());
            throw new RuntimeException(String.format("Failed to freeSampler %s: %s", resourceName.toLowerCase(), e.getMessage()), e);
        }
    }

    /**
     * Creates a summary of special tokens using LlamaVocabularyBinding
     */
    private String getSpecialTokensSummary(LlamaVocabulary vocabulary) {
        StringBuilder summary = new StringBuilder("Special Tokens Summary:\n");

        // BOS token
        Optional<Integer> bos = vocabularyService.getBosToken(vocabulary);
        if (bos.isPresent()) {
            String bosText = vocabularyService.getTokenText(vocabulary, bos.get());
            summary.append(String.format("  BOS: %d ('%s')\n", bos.get(), bosText != null ? bosText : "?"));
        } else {
            summary.append("  BOS: not available\n");
        }

        // EOS tokens
        List<Integer> eosTokens = getEndOfSequenceTokens(vocabulary);
        if (!eosTokens.isEmpty()) {
            summary.append("  EOS tokens:\n");
            for (Integer eos : eosTokens) {
                String eosText = vocabularyService.getTokenText(vocabulary, eos);
                summary.append(String.format("    %d ('%s')\n", eos, eosText != null ? eosText : "?"));
            }
        } else {
            summary.append("  EOS: not available\n");
        }

        // PAD token
        Optional<Integer> pad = vocabularyService.getPadToken(vocabulary);
        if (pad.isPresent()) {
            String padText = vocabularyService.getTokenText(vocabulary, pad.get());
            summary.append(String.format("  PAD: %d ('%s')\n", pad.get(), padText != null ? padText : "?"));
        } else {
            summary.append("  PAD: not available\n");
        }

        // Auto-add settings
        summary.append(String.format("  Auto-add BOS: %s\n", vocabularyService.getAddBos(vocabulary)));
        summary.append(String.format("  Auto-add EOS: %s", vocabularyService.getAddEos(vocabulary)));

        return summary.toString();
    }

    /**
     * Replacement for LlamaSpecialTokensService.getEndOfSequenceTokens()
     * Gets all end-of-sequence tokens using LlamaVocabularyBinding
     */
    private List<Integer> getEndOfSequenceTokens(LlamaVocabulary vocabulary) {
        List<Integer> eosTokens = new ArrayList<>();

        // Primary EOS token
        vocabularyService.getEosToken(vocabulary).ifPresent(eosTokens::add);

        // End-of-Text token (different from EOS in some models)
        vocabularyService.getEotToken(vocabulary).ifPresent(eot -> {
            if (!eosTokens.contains(eot)) {
                eosTokens.add(eot);
            }
        });

        return eosTokens;
    }

    /**
     * Checks if a token is a legitimate end-of-generation token
     */
    private boolean isLegitimateEndOfGeneration(LlamaVocabulary vocabulary, int tokenId) {
        // Check if it's marked as end-of-generation by the library
        boolean isEog = vocabularyService.isEndOfGeneration(vocabulary, tokenId);
        if (!isEog) {
            return false;
        }

        // Get all legitimate EOS tokens
        List<Integer> legitimateEosTokens = getEndOfSequenceTokens(vocabulary);

        // Only consider it legitimate if it's in our list of known EOS tokens
        boolean isLegitimate = legitimateEosTokens.contains(tokenId);

        if (!isLegitimate) {
            // Get token text for better debugging
            String tokenText = vocabularyService.getTokenText(vocabulary, tokenId);

            // Additional heuristics to filter out false positives
            if (tokenText != null && !tokenText.isEmpty()) {
                // Common false positives that should not end generation
                boolean isFalsePositive =
                    tokenText.matches("\\w+") ||                // Regular ASCII words
                    tokenText.matches("\\s+") ||                // Whitespace only
                    tokenText.matches("\\p{L}+") ||             // Unicode letters
                    tokenText.matches("\\p{N}+") ||             // Unicode numbers
                    tokenText.matches("[.,!?;:]") ||            // Common punctuation
                    tokenText.length() > 10;                    // Very long tokens are usually not EOS

                if (isFalsePositive) {
                    logger.debug("Filtering out false EOS positive - token {}: '{}'", tokenId,
                               tokenText.replace("\n", "\\n").replace("\r", "\\r"));
                    return false;
                }
            }

            logger.warn("Unknown EOS token {} not in legitimate list {}, token text: '{}'",
                       tokenId, legitimateEosTokens,
                       tokenText != null ? tokenText : "unknown");
        }

        return isLegitimate;
    }

    private static class CommandLineArgs {
        String modelPath = "";
        int nCtx = DEFAULT_N_CTX;
        int nGpuLayers = DEFAULT_N_GPU_LAYERS;
    }

}


