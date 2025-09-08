package com.quasarbyte.llama.cpp.jna.model.library;

/**
 * Represents a logit bias entry for biasing token probabilities.
 * <p>
 * Logit bias allows increasing or decreasing the likelihood of specific
 * tokens during sampling by applying a bias value to their logits.
 */
public class LlamaLogitBias {

    private final int token;
    private final float bias;

    /**
     * Creates a new logit bias entry.
     *
     * @param token the token ID to bias
     * @param bias the bias value (positive increases probability, negative decreases)
     */
    public LlamaLogitBias(int token, float bias) {
        this.token = token;
        this.bias = bias;
    }

    /**
     * Gets the token ID.
     *
     * @return the token ID
     */
    public int getToken() {
        return token;
    }

    /**
     * Gets the bias value.
     *
     * @return the bias value
     */
    public float getBias() {
        return bias;
    }

    @Override
    public String toString() {
        return "LlamaLogitBias{" +
                "token=" + token +
                ", bias=" + bias +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LlamaLogitBias that = (LlamaLogitBias) o;

        if (token != that.token) return false;
        return Float.compare(that.bias, bias) == 0;
    }

    @Override
    public int hashCode() {
        int result = token;
        result = 31 * result + (bias != +0.0f ? Float.floatToIntBits(bias) : 0);
        return result;
    }
}