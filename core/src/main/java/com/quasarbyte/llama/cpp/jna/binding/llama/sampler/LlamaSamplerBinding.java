package com.quasarbyte.llama.cpp.jna.binding.llama.sampler;

import com.quasarbyte.llama.cpp.jna.library.declaration.UInt32;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaSampler;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaSamplerChainParamsNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaTokenDataArray;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaVocabulary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaLogitBias;
import com.sun.jna.Pointer;

import java.util.List;

/**
 * Service interface for LLaMA sampling operations.
 * <p>
 * Sampling API Usage Example:
 * <pre>{@code
 * // prepare the sampling chain at the start
 * LlamaSamplerChainParamsNative sparams = llamaLibrary.llama_sampler_chain_default_params();
 *
 * LlamaSampler smpl = samplerService.createChainSampler(sparams);
 *
 * samplerService.addSamplerToChain(samplerService.createTopKSampler(50), smpl);
 * samplerService.addSamplerToChain(samplerService.createTopPSampler(0.9f, 1), smpl);
 * samplerService.addSamplerToChain(samplerService.createTemperatureSampler(0.8f), smpl);
 *
 * // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
 * // this sampler will be responsible to select the actual token
 * samplerService.addSamplerToChain(samplerService.createDistributionSampler(seed), smpl);
 *
 * // ...
 *
 * // decoding loop:
 * while (...) {
 *     // ...
 *
 *     processingService.decodeBatch(ctx, batch);
 *
 *     // sample from the logits of the last token in the batch
 *     int token = samplerService.sampleToken(smpl, ctx, -1);
 *
 *     // accept the token to update sampler state
 *     samplerService.acceptToken(smpl, token);
 * }
 *
 * samplerService.freeSampler(smpl);
 * }</pre>
 */
public interface LlamaSamplerBinding {
    // Chain Management
    LlamaSampler createChainSampler(LlamaSamplerChainParamsNative params);

    /**
     * @param disablePerformanceTimingMeasurements - Disable performance timing measurements for sampling operations.
     * @return sampler.
     */
    LlamaSampler createChainSampler(boolean disablePerformanceTimingMeasurements);

    void addSamplerToChain(LlamaSampler sourceSampler, LlamaSampler destinationSampler);

    LlamaSampler getChainSampler(LlamaSampler chain, int index);

    int getChainSamplerCount(LlamaSampler chain);

    LlamaSampler removeChainSampler(LlamaSampler chain, int index);

    // Basic Samplers  
    LlamaSampler createGreedySampler();

    LlamaSampler createDistributionSampler(UInt32 seed);

    // Advanced Samplers
    LlamaSampler createTopKSampler(int k);

    LlamaSampler createTopPSampler(float p, long minKeep);

    LlamaSampler createMinPSampler(float p, long minKeep);

    LlamaSampler createTypicalSampler(float p, long minKeep);

    LlamaSampler createTemperatureSampler(float temperature);

    LlamaSampler createTempExtSampler(float temperature, float delta, float exponent);

    LlamaSampler createXtcSampler(float p, float temperature, long minKeep, UInt32 seed);

    LlamaSampler createTopNSigmaSampler(float nSigma);

    LlamaSampler createMirostatSampler(int nVocab, int seed, float tau, float eta, UInt32 m);

    LlamaSampler createMirostatV2Sampler(UInt32 seed, float tau, float eta);

    // Grammar-based Samplers
    LlamaSampler createGrammarSampler(LlamaVocabulary vocabulary, String grammarStr, String grammarRoot);

    LlamaSampler createGrammarLazyPatternsSampler(LlamaVocabulary vocabulary, String grammarStr, String grammarRoot,
                                                  String[] triggerPatterns, int[] triggerTokens);

    // Penalty Samplers
    LlamaSampler createPenaltiesSampler(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent);

    LlamaSampler createDontRepeatYourselfSampler(LlamaVocabulary vocabulary, int nCtxTrain, float dryMultiplier, float dryBase,
                                                 int dryAllowedLength, int dryPenaltyLastN, String[] seqBreakers);

    LlamaSampler createLogitBiasSampler(int nVocab, List<LlamaLogitBias> logitBias);

    LlamaSampler createInfillSampler(LlamaVocabulary vocabulary);

    // Sampler Operations
    LlamaSampler initializeSampler(Pointer samplerInterface, LlamaContext context);

    String getSamplerName(LlamaSampler sampler);

    void acceptToken(LlamaSampler sampler, int token);

    void applySampler(LlamaSampler sampler, LlamaTokenDataArray probabilities);

    void resetSampler(LlamaSampler sampler);

    LlamaSampler cloneSampler(LlamaSampler sampler);

    UInt32 getSamplerSeed(LlamaSampler sampler);

    // Sampling
    int sampleToken(LlamaSampler sampler, LlamaContext context, int index);

    void printPerformance(LlamaSampler sampler);

    void freeSampler(LlamaSampler sampler);
}
