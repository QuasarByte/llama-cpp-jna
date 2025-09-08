package com.quasarbyte.llama.cpp.jna.binding.llama.sampler;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallLongResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.UInt32;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Objects;

public class LlamaSamplerBindingImpl implements LlamaSamplerBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaSamplerBindingImpl.class);
    private final LlamaLibrary llamaLibrary;

    public LlamaSamplerBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public LlamaSampler createChainSampler(LlamaSamplerChainParamsNative params) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_chain_init(params);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create sampler chain");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer).setChainRole(ChainRole.HEAD);
    }

    @Override
    public LlamaSampler createChainSampler(boolean disablePerformanceTimingMeasurements) {
        try {
            LlamaSamplerChainParamsNative params = llamaLibrary.llama_sampler_chain_default_params();
            params.no_perf = (byte) (disablePerformanceTimingMeasurements ? 1 : 0);
            return createChainSampler(params);
        } catch (Exception e) {
            throw new LlamaCppJnaException(String.format("Failed to create sampler chain, error: '%s'", e.getMessage()), e);
        }
    }

    @Override
    public LlamaSampler createGreedySampler() {
        final LlamaSamplerNative samplerPointer;

        try {
            samplerPointer = llamaLibrary.llama_sampler_init_greedy();
        } catch (Exception e) {
            throw new LlamaCppJnaException(String.format("Failed to create greedy sampler, error: '%s'", e.getMessage()));
        }

        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create greedy sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createMinPSampler(float p, long minKeep) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_min_p(p, minKeep);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create min-p sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createTemperatureSampler(float temperature) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_temp(temperature);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create temperature sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createDistributionSampler(UInt32 seed) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_dist(seed);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create distribution sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public void addSamplerToChain(LlamaSampler sourceSampler, LlamaSampler destinationSampler) {
        Objects.requireNonNull(sourceSampler);
        Objects.requireNonNull(sourceSampler.getSamplerPointer());
        Objects.requireNonNull(destinationSampler);
        Objects.requireNonNull(destinationSampler.getSamplerPointer());

        if (ChainRole.HEAD.equals(sourceSampler.getChainRole())) {
            throw new LlamaCppJnaException("Source sampler can not be head of chain.");
        }

        if (ChainRole.NODE.equals(sourceSampler.getChainRole())) {
            throw new LlamaCppJnaException("Source sampler already added to a chain.");
        }

        if (!ChainRole.HEAD.equals(destinationSampler.getChainRole())) {
            throw new LlamaCppJnaException("Destination sample can be only head of chain");
        }

        if (sourceSampler == destinationSampler) {
            throw new LlamaCppJnaException("Source and destination sampler are the same");
        }

        if (sourceSampler.getSamplerPointer() == destinationSampler.getSamplerPointer()) {
            throw new LlamaCppJnaException("Source and destination samplers pointers are the same");
        }

        llamaLibrary.llama_sampler_chain_add(destinationSampler.getSamplerPointer(), sourceSampler.getSamplerPointer());
    }

    @Override
    public int sampleToken(LlamaSampler sampler, LlamaContext context, int index) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());
        return llamaLibrary.llama_sampler_sample(sampler.getSamplerPointer(), context.getContextPointer(), index);
    }

    @Override
    public void printPerformance(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());
        llamaLibrary.llama_perf_sampler_print(sampler.getSamplerPointer());
    }

    @Override
    public void freeSampler(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());

        if (ChainRole.NODE.equals(sampler.getChainRole())) {
            throw new LlamaCppJnaException("Sampler included into chain can not be freed directly. Use head of chain.");
        }

        llamaLibrary.llama_sampler_free(sampler.getSamplerPointer());
    }

    @Override
    public LlamaSampler getChainSampler(LlamaSampler sampler, int index) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());

        if (!ChainRole.HEAD.equals(sampler.getChainRole())) {
            throw new LlamaCppJnaException("Not a chain sampler");
        }

        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_chain_get(sampler.getSamplerPointer(), index);

        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to get a chain sampler");
        }

        return new LlamaSampler().setSamplerPointer(samplerPointer).setChainRole(ChainRole.NODE);
    }

    @Override
    public int getChainSamplerCount(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());

        if (!ChainRole.HEAD.equals(sampler.getChainRole())) {
            throw new LlamaCppJnaException("Not a chain sampler");
        }
        return llamaLibrary.llama_sampler_chain_n(sampler.getSamplerPointer());
    }

    @Override
    public LlamaSampler removeChainSampler(LlamaSampler sampler, int index) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());

        if (!ChainRole.HEAD.equals(sampler.getChainRole())) {
            throw new LlamaCppJnaException("Not a chain sampler");
        }
        LlamaSamplerNative removedPointer = llamaLibrary.llama_sampler_chain_remove(sampler.getSamplerPointer(), index);
        if (removedPointer == null) {
            throw new LlamaCppJnaException("Failed to create a chain sampler");
        }
        return new LlamaSampler().setSamplerPointer(removedPointer).setChainRole(null);
    }

    @Override
    public LlamaSampler createTopKSampler(int k) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_top_k(k);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create top-k sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createTopPSampler(float p, long minKeep) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_top_p(p, minKeep);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create top-p sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createTypicalSampler(float p, long minKeep) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_typical(p, minKeep);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create typical sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createTempExtSampler(float temperature, float delta, float exponent) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_temp_ext(temperature, delta, exponent);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create extended temperature sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createXtcSampler(float p, float temperature, long minKeep, UInt32 seed) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_xtc(p, temperature, minKeep, seed);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create XTC sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createTopNSigmaSampler(float nSigma) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_top_n_sigma(nSigma);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create top-n-sigma sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createMirostatSampler(int nVocab, int seed, float tau, float eta, UInt32 m) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_mirostat(nVocab, seed, tau, eta, m);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create mirostat sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createMirostatV2Sampler(UInt32 seed, float tau, float eta) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_mirostat_v2(seed, tau, eta);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create mirostat v2 sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createGrammarSampler(LlamaVocabulary vocabulary, String grammarStr, String grammarRoot) {
        Objects.requireNonNull(vocabulary);
        Objects.requireNonNull(vocabulary.getVocabularyPointer());
        Objects.requireNonNull(grammarStr);
        Objects.requireNonNull(grammarRoot);

        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_grammar(vocabulary.getVocabularyPointer(), grammarStr, grammarRoot);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create grammar sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createGrammarLazyPatternsSampler(LlamaVocabulary vocabulary,
                                                         String grammarStr,
                                                         String grammarRoot,
                                                         String[] triggerPatterns,
                                                         int[] triggerTokens) {
        Objects.requireNonNull(vocabulary);
        Objects.requireNonNull(vocabulary.getVocabularyPointer());
        Objects.requireNonNull(grammarStr);
        Objects.requireNonNull(grammarRoot);
        Objects.requireNonNull(triggerPatterns);
        Objects.requireNonNull(triggerTokens);

        try {
            LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_grammar_lazy_patterns(
                    vocabulary.getVocabularyPointer(),
                    grammarStr,
                    grammarRoot,
                    triggerPatterns,
                    triggerPatterns.length,
                    triggerTokens,
                    triggerTokens.length);

            if (samplerPointer == null) {
                throw new LlamaCppJnaException("Failed to create lazy grammar sampler");
            }

            return new LlamaSampler().setSamplerPointer(samplerPointer);
        } catch (Exception e) {
            logger.error("Error creating lazy grammar sampler", e);
            throw new LlamaCppJnaException("Failed to create lazy grammar sampler", e);
        }
    }

    @Override
    public LlamaSampler createPenaltiesSampler(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent) {
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent);
        if (samplerPointer == null) {
            throw new LlamaCppJnaException("Failed to create penalties sampler");
        }
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler createDontRepeatYourselfSampler(LlamaVocabulary vocabulary,
                                                        int nCtxTrain,
                                                        float dryMultiplier,
                                                        float dryBase,
                                                        int dryAllowedLength,
                                                        int dryPenaltyLastN,
                                                        String[] seqBreakers) {
        Objects.requireNonNull(vocabulary);
        Objects.requireNonNull(vocabulary.getVocabularyPointer());
        Objects.requireNonNull(seqBreakers);

        try {

            LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_dry(
                    vocabulary.getVocabularyPointer(),
                    nCtxTrain,
                    dryMultiplier,
                    dryBase,
                    dryAllowedLength,
                    dryPenaltyLastN,
                    seqBreakers,
                    seqBreakers.length);

            if (samplerPointer == null) {
                throw new LlamaCppJnaException("Failed to create DRY sampler");
            }

            return new LlamaSampler().setSamplerPointer(samplerPointer);
        } catch (Exception e) {
            logger.error("Error creating DRY sampler", e);
            throw new LlamaCppJnaException("Failed to create DRY sampler", e);
        }
    }

    @Override
    public LlamaSampler createLogitBiasSampler(int nVocab, List<LlamaLogitBias> logitBias) {
        Objects.requireNonNull(logitBias);

        // Create contiguous array using JNA's array allocation
        LlamaLogitBiasNative[] biasNatives;
        if (logitBias.isEmpty()) {
            biasNatives = new LlamaLogitBiasNative[0];
        } else {
            LlamaLogitBiasNative firstBias = new LlamaLogitBiasNative();
            biasNatives = (LlamaLogitBiasNative[]) firstBias.toArray(logitBias.size());

            for (int i = 0; i < biasNatives.length; i++) {
                LlamaLogitBias source = logitBias.get(i);
                biasNatives[i].token = source.getToken();
                biasNatives[i].bias = source.getBias();
            }
        }

        try {
            LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_logit_bias(nVocab, biasNatives.length, biasNatives);

            if (samplerPointer == null) {
                throw new LlamaCppJnaException("Failed to create logit bias sampler");
            }

            return new LlamaSampler().setSamplerPointer(samplerPointer);

        } catch (Exception e) {
            logger.error("Error creating logit bias sampler", e);
            throw new LlamaCppJnaException("Failed to create logit bias sampler", e);
        }
    }

    @Override
    public LlamaSampler createInfillSampler(LlamaVocabulary vocabulary) {
        Objects.requireNonNull(vocabulary);
        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init_infill(vocabulary.getVocabularyPointer());
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public LlamaSampler initializeSampler(Pointer samplerInterface, LlamaContext context) {
        Objects.requireNonNull(samplerInterface);

        LlamaSamplerNative samplerPointer = llamaLibrary.llama_sampler_init(samplerInterface, context.getContextPointer());
        return new LlamaSampler().setSamplerPointer(samplerPointer);
    }

    @Override
    public String getSamplerName(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        return llamaLibrary.llama_sampler_name(sampler.getSamplerPointer());
    }

    @Override
    public void acceptToken(LlamaSampler sampler, int token) {
        Objects.requireNonNull(sampler);
        llamaLibrary.llama_sampler_accept(sampler.getSamplerPointer(), token);
    }

    @Override
    public void applySampler(LlamaSampler sampler, LlamaTokenDataArray probabilities) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(probabilities);
        LlamaTokenDataArrayNative nativeArray = probabilities.toNative();
        llamaLibrary.llama_sampler_apply(sampler.getSamplerPointer(), nativeArray);
    }

    @Override
    public void resetSampler(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        llamaLibrary.llama_sampler_reset(sampler.getSamplerPointer());
    }

    @Override
    public LlamaSampler cloneSampler(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());
        LlamaSamplerNative clonedPointer = llamaLibrary.llama_sampler_clone(sampler.getSamplerPointer());
        return new LlamaSampler().setSamplerPointer(clonedPointer).setChainRole(sampler.getChainRole());
    }

    @Override
    public UInt32 getSamplerSeed(LlamaSampler sampler) {
        Objects.requireNonNull(sampler);
        Objects.requireNonNull(sampler.getSamplerPointer());

        UInt32 seedValue = llamaLibrary.llama_sampler_get_seed(sampler.getSamplerPointer());

        if (seedValue.longValue() == -1) {
            logger.error("Failed to get sampler seed for sampler, seed values is not applicable, seed value: {}", sampler.getSamplerPointer());
            throw  new LlamaFunctionCallLongResultException(seedValue.longValue(), String.format( "Failed to get sampler seed for sampler, seed values is not applicable, seed value: %d", seedValue.longValue()));
        }

        return seedValue;
    }
}
