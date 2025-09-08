package com.quasarbyte.llama.cpp.jna.examples.simple;

import com.quasarbyte.llama.cpp.jna.binding.llama.processing.LlamaProcessingBinding;
import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibraryFactory;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibraryFactory;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader.GgmlBackendLoader;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader.GgmlBackendLoaderFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.backend.LlamaBackendBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.backend.LlamaBackendBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.batch.LlamaBatchBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.batch.LlamaBatchBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.context.LlamaContextBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.context.LlamaContextBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.model.LlamaModelBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.model.LlamaModelBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.performance.LlamaPerformanceBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.performance.LlamaPerformanceBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.processing.LlamaProcessingBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.sampler.LlamaSamplerBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.sampler.LlamaSamplerBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.token.LlamaTokenBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.token.LlamaTokenBindingFactory;
import com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary.LlamaVocabularyBinding;
import com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary.LlamaVocabularyBindingFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.nio.file.Paths;

import static com.quasarbyte.llama.cpp.jna.binding.llama.token.LlamaTokenBindingConstants.EOS_TOKEN_ID;

public class Simple {
    private static final Logger logger = LoggerFactory.getLogger(Simple.class);
    private static final String DEFAULT_PROMPT = "Hello my name is";
    private static final int DEFAULT_N_PREDICT = 32;
    private static final int DEFAULT_N_GPU_LAYERS = 0;
    private static final int TOKEN_TO_PIECE_MAX_LENGTH = 128;

    private static void printUsage() {
        System.out.println();
        System.out.println("example usage:");
        System.out.println();
        System.out.printf("  %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]", Simple.class.getName());
        System.out.println();
    }

    public static void main(String[] args) {
        Simple simple = new Simple();
        simple.run(args);
    }

    private void run(String[] args) {
        GgmlLibrary ggmlLibrary = new GgmlLibraryFactory().getInstance();
        GgmlBackendLoader backendLoaderService = new GgmlBackendLoaderFactory().create(ggmlLibrary);

        LlamaLibrary llamaLibrary = new LlamaLibraryFactory().getInstance();
        LlamaBackendBinding llamaBackendBinding = new LlamaBackendBindingFactory().create(llamaLibrary);

        LlamaModelBinding modelLoaderService = new LlamaModelBindingFactory().create(llamaLibrary);

        LlamaVocabularyBinding llamaVocabularyBinding = new LlamaVocabularyBindingFactory().create(llamaLibrary);

        LlamaTokenBinding tokenService = new LlamaTokenBindingFactory().create(llamaLibrary);
        LlamaBatchBinding batchService = new LlamaBatchBindingFactory().create(llamaLibrary);
        LlamaContextBinding contextService = new LlamaContextBindingFactory().create(llamaLibrary);
        LlamaSamplerBinding samplerService = new LlamaSamplerBindingFactory().create(llamaLibrary);
        LlamaProcessingBinding llamaProcessingBinding = new LlamaProcessingBindingFactory().create(llamaLibrary);

        LlamaPerformanceBinding llamaPerformanceBinding = new LlamaPerformanceBindingFactory().create(llamaLibrary);

        CommandLineArgs cmdArgs = parseCommandLineArgs(args);
        if (cmdArgs.modelPath.isEmpty()) {
            printUsage();
            System.exit(1);
        }

        try {
            logger.info("Starting llama.cpp initialization...");

            backendLoaderService.loadBackend();
            llamaBackendBinding.backendInit();

            LlamaModel model = loadModel(modelLoaderService, Paths.get(cmdArgs.modelPath), cmdArgs.nGpuLayers);

            LlamaVocabulary vocabulary = llamaVocabularyBinding.getVocabulary(model);

            int[] promptTokens = tokenizePrompt(tokenService, model, vocabulary, cmdArgs.prompt);

            int nCtx = promptTokens.length + cmdArgs.nPredict - 1;
            int nBatch = promptTokens.length;
            LlamaContext context = createContext(contextService, model, nCtx, nBatch);

            LlamaSampler sampler = createSampler(samplerService);

            LlamaBatch batch = createBatch(batchService, promptTokens);

            printPromptFromTokens(tokenService, model, vocabulary, promptTokens);

            GenerationResult result = runGenerationLoop(llamaLibrary, llamaProcessingBinding, samplerService, batchService, tokenService,
                    model, vocabulary, context, sampler, batch, promptTokens.length, cmdArgs.nPredict);

            printPerformanceMetrics(samplerService, contextService, llamaPerformanceBinding, sampler, context, result);

            cleanup(batchService, samplerService, contextService, modelLoaderService, llamaBackendBinding,
                    result.finalBatch, sampler, context, model);

        } catch (Exception e) {
            logger.error("Error: {}", e.getMessage(), e);
            System.err.println(e.getMessage());
            System.exit(1);
        }
    }

    private CommandLineArgs parseCommandLineArgs(String[] args) {
        CommandLineArgs cmdArgs = new CommandLineArgs();

        int i = 0;
        while (i < args.length) {
            if ("-m".equals(args[i])) {
                if (i + 1 < args.length) {
                    cmdArgs.modelPath = args[++i];
                } else {
                    printUsage();
                    System.exit(1);
                }
            } else if ("-n".equals(args[i])) {
                if (i + 1 < args.length) {
                    try {
                        cmdArgs.nPredict = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException e) {
                        printUsage();
                        System.exit(1);
                    }
                } else {
                    printUsage();
                    System.exit(1);
                }
            } else if ("-ngl".equals(args[i])) {
                if (i + 1 < args.length) {
                    try {
                        cmdArgs.nGpuLayers = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException e) {
                        printUsage();
                        System.exit(1);
                    }
                } else {
                    printUsage();
                    System.exit(1);
                }
            } else {
                StringBuilder promptBuilder = new StringBuilder(args[i++]);
                while (i < args.length) {
                    promptBuilder.append(" ").append(args[i++]);
                }
                cmdArgs.prompt = promptBuilder.toString();
                break;
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

    private int[] tokenizePrompt(LlamaTokenBinding tokenService,
                                 LlamaModel model,
                                 LlamaVocabulary vocabulary,
                                 String prompt) {
        logger.debug("Tokenizing prompt: \"{}\"", prompt);
        try {
            int[] promptTokens = tokenService.tokenize(model, vocabulary, prompt, true, true);
            logger.debug("Prompt tokenized into {} tokens", promptTokens.length);

            for (int j = 0; j < Math.min(5, promptTokens.length); j++) {
                logger.debug("Token[{}] = {}", j, promptTokens[j]);
            }
            return promptTokens;
        } catch (Exception e) {
            logger.error("Failed to tokenize: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to tokenize: %s", e.getMessage()), e);
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

    private LlamaSampler createSampler(LlamaSamplerBinding samplerService) {
        logger.debug("Creating sampler...");
        try {
            LlamaSampler sampler = samplerService.createChainSampler(false);
            LlamaSampler greedy = samplerService.createGreedySampler();
            samplerService.addSamplerToChain(greedy, sampler);
            logger.debug("Sampler created successfully");
            return sampler;
        } catch (Exception e) {
            logger.error("Failed to create sampler: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to create sampler: %s", e.getMessage()), e);
        }
    }

    private LlamaBatch createBatch(LlamaBatchBinding batchService,
                                   int[] promptTokens) {
        logger.debug("Creating batch...");
        try {
            LlamaBatch batch = batchService.createBatch(promptTokens);
            logger.debug("Batch created with {} tokens", batch.getTokenCount());
            return batch;
        } catch (Exception e) {
            logger.error("Failed to create batch: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to create batch: %s", e.getMessage()), e);
        }
    }

    private void printPromptFromTokens(LlamaTokenBinding tokenService,
                                       LlamaModel model,
                                       LlamaVocabulary vocabulary,
                                       int[] promptTokens) {
        logger.debug("Reconstructing prompt from tokens:");
        for (int token : promptTokens) {
            try {
                String piece = tokenService.tokenToPiece(model, vocabulary, token, true, TOKEN_TO_PIECE_MAX_LENGTH);
                System.out.print(piece);
            } catch (Exception e) {
                logger.error("Failed to convert token {} to piece: {}", token, e.getMessage());
                throw new RuntimeException(String.format("Failed to convert token %d to piece: %s", token, e.getMessage()), e);
            }
        }
        System.out.println();
    }

    private GenerationResult runGenerationLoop(LlamaLibrary llamaLibrary,
                                               LlamaProcessingBinding llamaProcessingBinding,
                                               LlamaSamplerBinding samplerService,
                                               LlamaBatchBinding batchService,
                                               LlamaTokenBinding tokenService,
                                               LlamaModel model,
                                               LlamaVocabulary vocabulary,
                                               LlamaContext context,
                                               LlamaSampler sampler,
                                               LlamaBatch batch,
                                               int promptLength,
                                               int nPredict) {
        logger.info("Starting generation loop...");
        long startTime = llamaLibrary.llama_time_us();
        int nDecode = 0;
        LlamaBatch currentBatch = batch;

        for (int nPos = 0; nPos + currentBatch.getTokenCount() < promptLength + nPredict; ) {
            logger.debug("Position {}, batch size {}", nPos, currentBatch.getTokenCount());

            if (!decodeBatch(llamaProcessingBinding, context, currentBatch)) {
                break;
            }

            nPos += currentBatch.getTokenCount();

            int newTokenId = sampleNextToken(samplerService, sampler, context);
            if (newTokenId == -1) {
                break;
            }

            boolean isEndOfGeneration = checkEndOfGeneration(tokenService, vocabulary, newTokenId);

            printToken(tokenService, model, vocabulary, newTokenId);

            if (isEndOfGeneration) {
                break;
            }

            LlamaBatch nextBatch = createNextBatch(batchService, currentBatch, newTokenId);
            if (nextBatch == null) {
                break;
            }

            currentBatch = nextBatch;
            nDecode++;
        }

        System.out.println();
        long endTime = llamaLibrary.llama_time_us();

        return new GenerationResult(nDecode, startTime, endTime, currentBatch);
    }

    private boolean decodeBatch(LlamaProcessingBinding llamaProcessingBinding,
                                LlamaContext context,
                                LlamaBatch batch) {
        try {
            int result = llamaProcessingBinding.decodeBatch(context, batch);
            if (result != 0) {
                logger.warn("Decode returned non-zero: {}", result);
                return false;
            }
            logger.debug("Decode successful");
            return true;
        } catch (Exception e) {
            logger.error("Decode failed: {}", e.getMessage());
            throw new RuntimeException(String.format("Decode failed: %s", e.getMessage()), e);
        }
    }

    private int sampleNextToken(LlamaSamplerBinding samplerService,
                                LlamaSampler sampler,
                                LlamaContext context) {
        try {
            int newTokenId = samplerService.sampleToken(sampler, context, -1);
            logger.debug("Sampled token: {}", newTokenId);
            return newTokenId;
        } catch (Exception e) {
            logger.error("Sampling failed: {}", e.getMessage());
            throw new RuntimeException(String.format("Sampling failed: %s", e.getMessage()), e);
        }
    }

    private boolean checkEndOfGeneration(LlamaTokenBinding tokenService,
                                         LlamaVocabulary vocabulary,
                                         int tokenId) {
        try {
            boolean isEOG = tokenService.isEndOfGeneration(vocabulary, tokenId);
            if (isEOG) {
                logger.debug("End of generation token detected (token {})", tokenId);
                isEOG = (tokenId == EOS_TOKEN_ID);
                if (!isEOG) {
                    logger.debug("Ignoring EOG detection for token {} - continuing generation", tokenId);
                }
            }
            return isEOG;
        } catch (Exception e) {
            logger.error("Failed to check EOG: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to check EOG: %s", e.getMessage()), e);
        }
    }

    private void printToken(LlamaTokenBinding tokenService,
                            LlamaModel model,
                            LlamaVocabulary vocabulary,
                            int tokenId) {
        try {
            String piece = tokenService.tokenToPiece(model, vocabulary, tokenId, true, TOKEN_TO_PIECE_MAX_LENGTH);
            System.out.print(piece);
            System.out.flush();
            logger.debug("Generated piece: '{}'", piece);
        } catch (Exception e) {
            logger.error("Failed to convert new token to piece: {}", e.getMessage());
            throw new RuntimeException(String.format("Failed to convert new token to piece: %s", e.getMessage()), e);
        }
    }

    private LlamaBatch createNextBatch(LlamaBatchBinding batchService,
                                       LlamaBatch currentBatch,
                                       int newTokenId) {
        try {
            batchService.freeBatch(currentBatch);
            return batchService.createBatch(new int[]{newTokenId});
        } catch (Exception e) {
            logger.error("Failed to create next batch: {}", e.getMessage());
            throw new RuntimeException("Failed to create next batch: " + e.getMessage());
        }
    }

    private void printPerformanceMetrics(LlamaSamplerBinding samplerService,
                                         LlamaContextBinding contextService,
                                         LlamaPerformanceBinding llamaPerformanceBinding,
                                         LlamaSampler sampler,
                                         LlamaContext context,
                                         GenerationResult result) {
        logger.info("Decoded {} tokens in {:.2f} s, speed: {:.2f} t/s",
                result.nDecode, result.getDurationSeconds(), result.getTokensPerSecond());

        logger.debug("Printing performance metrics...");
        try {
            samplerService.printPerformance(sampler);
            contextService.printPerformance(context);

            LlamaPerformanceContextDataNative llamaPerformanceContextDataNative = llamaPerformanceBinding.getContextPerformance(context);
            logger.debug("Printing performance metrics: {}", llamaPerformanceContextDataNative);
            logger.debug("Printing performance metrics successful");
        } catch (Exception e) {
            logger.error("Failed to print performance: {}", e.getMessage());
            throw new RuntimeException("Failed to print performance: " + e.getMessage());
        }
    }

    private void cleanup(LlamaBatchBinding batchService,
                         LlamaSamplerBinding samplerService,
                         LlamaContextBinding contextService,
                         LlamaModelBinding modelLoaderService,
                         LlamaBackendBinding llamaBackendBinding,
                         LlamaBatch batch,
                         LlamaSampler sampler,
                         LlamaContext context,
                         LlamaModel model) {
        logger.debug("Starting cleanup...");

        try {
            safelyFree("Batch", () -> batchService.freeBatch(batch));
            safelyFree("Sampler", () -> samplerService.freeSampler(sampler));
            safelyFree("Context", () -> contextService.freeContext(context));
            safelyFree("Model", () -> modelLoaderService.freeModel(model));
            safelyFree("Backend", llamaBackendBinding::freeBackend);
        } catch (Exception e) {
            logger.error("Failed to cleanup: {}", e.getMessage());
            throw new RuntimeException("Failed to cleanup: " + e.getMessage());
        }

        logger.debug("Cleanup completed successfully");
    }

    private void safelyFree(String resourceName, Runnable freeOperation) {
        try {
            freeOperation.run();
            logger.debug("{} freed", resourceName);
        } catch (Exception e) {
            logger.error("Failed to freeSampler {}: {}", resourceName.toLowerCase(), e.getMessage());
            throw new RuntimeException(String.format("Failed to freeSampler %s: %s", resourceName.toLowerCase(), e.getMessage()), e);
        }
    }

    private static class CommandLineArgs {
        String modelPath = "";
        String prompt = DEFAULT_PROMPT;
        int nGpuLayers = DEFAULT_N_GPU_LAYERS;
        int nPredict = DEFAULT_N_PREDICT;
    }

    private static class GenerationResult {
        final int nDecode;
        final long startTime;
        final long endTime;
        final LlamaBatch finalBatch;

        GenerationResult(int nDecode, long startTime, long endTime, LlamaBatch finalBatch) {
            this.nDecode = nDecode;
            this.startTime = startTime;
            this.endTime = endTime;
            this.finalBatch = finalBatch;
        }

        double getDurationSeconds() {
            return (endTime - startTime) / 1000000.0;
        }

        double getTokensPerSecond() {
            return nDecode / getDurationSeconds();
        }
    }
}