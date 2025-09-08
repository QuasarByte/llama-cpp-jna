package com.quasarbyte.llama.cpp.jna.binding.llama.modelsplit;

/**
 * Service for handling model file splitting and path generation.
 * <p>
 * This binding provides utilities for working with split model files,
 * including generating split file paths and extracting prefixes from
 * split file names. This is useful for handling large models that are
 * distributed across multiple files.
 */
public interface LlamaModelSplitBinding {

    /**
     * Generate split file path for model sharding.
     * <p>
     * Creates a file path following the standard split naming convention:
     * {@code <prefix>-<split_no>-of-<split_count>.gguf}
     *
     * @param pathPrefix the base path/name prefix for split files
     * @param splitNumber the split number (0-based)
     * @param splitCount the total number of splits
     * @return the generated split file path, or null on error
     */
    String generateSplitPath(String pathPrefix, int splitNumber, int splitCount);

    /**
     * Generate split file path with maximum length limit.
     *
     * @param pathPrefix the base path/name prefix for split files
     * @param splitNumber the split number (0-based)
     * @param splitCount the total number of splits
     * @param maxLength maximum length of the generated path
     * @return the generated split file path (truncated if necessary), or null on error
     */
    String generateSplitPath(String pathPrefix, int splitNumber, int splitCount, int maxLength);

    /**
     * Extract split file prefix with maximum length limit.
     *
     * @param splitPath the split file path to parse
     * @param splitNumber the expected split number
     * @param splitCount the expected total split count
     * @param maxLength maximum length of the extracted prefix
     * @return the extracted prefix (truncated if necessary), or null if parsing fails
     */
    String extractSplitPrefix(String splitPath, int splitNumber, int splitCount, int maxLength);

    /**
     * Check if a file path follows the split file naming convention.
     *
     * @param filePath the file path to check
     * @return true if it follows the split naming convention, false otherwise
     */
    boolean isSplitFilePath(String filePath);

    /**
     * Parse split information from a split file path.
     *
     * @param splitPath the split file path to parse
     * @return split information (split number, total count), or null if not a split file
     */
    SplitInfo parseSplitInfo(String splitPath);

    /**
     * Information extracted from a split file path.
     */
    class SplitInfo {
        private final int splitNumber;
        private final int totalSplits;
        private final String prefix;

        public SplitInfo(int splitNumber, int totalSplits, String prefix) {
            this.splitNumber = splitNumber;
            this.totalSplits = totalSplits;
            this.prefix = prefix;
        }

        public int getSplitNumber() {
            return splitNumber;
        }

        public int getTotalSplits() {
            return totalSplits;
        }

        public String getPrefix() {
            return prefix;
        }

        @Override
        public String toString() {
            return "SplitInfo{" +
                    "splitNumber=" + splitNumber +
                    ", totalSplits=" + totalSplits +
                    ", prefix='" + prefix + '\'' +
                    '}';
        }
    }
}