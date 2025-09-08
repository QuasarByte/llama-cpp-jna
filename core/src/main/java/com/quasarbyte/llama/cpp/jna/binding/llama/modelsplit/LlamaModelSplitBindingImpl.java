package com.quasarbyte.llama.cpp.jna.binding.llama.modelsplit;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Implementation of LlamaModelSplitBinding for handling model file splitting.
 */
public class LlamaModelSplitBindingImpl implements LlamaModelSplitBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaModelSplitBindingImpl.class);
    private static final Pattern SPLIT_PATTERN = Pattern.compile("(.+)-(\\d{5})-of-(\\d{5})\\.gguf$");
    private static final int MAX_PATH_LENGTH = getSystemMaxPath();

    private final LlamaLibrary llamaLibrary;
    private final LlamaStringBufferReader llamaStringBufferReader;

    public LlamaModelSplitBindingImpl(LlamaLibrary llamaLibrary, LlamaStringBufferReader llamaStringBufferReader) {
        this.llamaLibrary = llamaLibrary;
        this.llamaStringBufferReader = llamaStringBufferReader;
    }

    @Override
    public String generateSplitPath(String pathPrefix, int splitNumber, int splitCount) {
        return generateSplitPath(pathPrefix, splitNumber, splitCount, MAX_PATH_LENGTH);
    }

    @Override
    public String generateSplitPath(String pathPrefix, int splitNumber, int splitCount, int maxLength) {

        if (pathPrefix == null) {
            logger.error("Parameter pathPrefix cannot be null");
            throw new LlamaCppJnaException("Parameter pathPrefix cannot be null");
        }

        if (splitNumber < 0) {
            logger.error("Parameter splitNumber can not be less than 0");
            throw new LlamaCppJnaException("Parameter splitNumber can not be less than 0");
        }

        if (splitCount < 0) {
            logger.error("Parameter splitCount can not be less than 0");
            throw new LlamaCppJnaException("Parameter splitCount can not be less than 0");
        }

        if (maxLength > MAX_PATH_LENGTH) {
            logger.error("Parameter maxLength can not be greater than MAX_PATH_LENGTH");
            throw new LlamaCppJnaException("Parameter maxLength can not be greater than MAX_PATH_LENGTH");
        }

        try {

            byte[] buffer = new byte[maxLength];
            int length = llamaLibrary.llama_split_path(buffer, buffer.length, pathPrefix, splitNumber, splitCount);

            if (length < 0) {
                logger.error("Failed to generate split path, pathPrefix: '{}', splitNumber: {}, splitCount: {}, maxLength: {}, length: {}", pathPrefix, splitNumber, splitCount, maxLength, length);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to generate split path, pathPrefix: '%s', splitNumber: %d, splitCount: %d, maxLength: %d, length: %d", pathPrefix, splitNumber, splitCount, maxLength, length));
            }

            return llamaStringBufferReader.readString(buffer, length);

        } catch (Exception e) {
            logger.error("Error generating split path, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error generating split path, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public String extractSplitPrefix(String splitPath, int splitNumber, int splitCount, int maxLength) {

        if (splitPath == null) {
            logger.error("Parameter pathPrefix cannot be null");
            throw new LlamaCppJnaException("Parameter pathPrefix cannot be null");
        }

        if (splitNumber < 0) {
            logger.error("Parameter splitNumber can not be less than 0");
            throw new LlamaCppJnaException("Parameter splitNumber can not be less than 0");
        }

        if (splitCount < 0) {
            logger.error("Parameter splitCount can not be less than 0");
            throw new LlamaCppJnaException("Parameter splitCount can not be less than 0");
        }

        if (maxLength < 0) {
            logger.error("Parameter maxLength can not be less than 0");
            throw new LlamaCppJnaException("Parameter maxLength can not be less than 0");
        }

        if (maxLength > MAX_PATH_LENGTH) {
            logger.error("Parameter maxLength can not be greater than MAX_PATH_LENGTH");
            throw new LlamaCppJnaException("Parameter maxLength can not be greater than MAX_PATH_LENGTH");
        }

        try {

            byte[] buffer = new byte[maxLength];
            int result = llamaLibrary.llama_split_prefix(buffer, maxLength, splitPath, splitNumber, splitCount);

            if (result < 0) {
                logger.error("Failed to generating split prefix, splitPath: '{}', splitNumber: {}, splitCount: {}, maxLength: {}, result: {}", splitPath, splitNumber, splitCount, maxLength, result);
                throw new LlamaFunctionCallIntResultException(result, String.format("Failed to generate split path, splitPath: '%s', splitNumber: %d, splitCount: %d, result: %d", splitPath, splitNumber, splitCount, result));
            }

            return llamaStringBufferReader.readString(buffer, result);

        } catch (Exception e) {
            logger.error("Error generating split prefix, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Error generating split prefix, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public boolean isSplitFilePath(String filePath) {

        if (filePath == null) {
            logger.error("Parameter filePath cannot be null");
            throw new LlamaCppJnaException("Parameter filePath cannot be null");
        }

        if (filePath.trim().isEmpty()) {
            logger.error("Parameter filePath cannot be blank");
            throw new LlamaCppJnaException("Parameter filePath cannot be blank");
        }

        return SPLIT_PATTERN.matcher(filePath).matches();
    }

    @Override
    public SplitInfo parseSplitInfo(String splitPath) {
        try {
            if (splitPath == null || splitPath.trim().isEmpty()) {
                return null;
            }

            Matcher matcher = SPLIT_PATTERN.matcher(splitPath);
            if (!matcher.matches()) {
                return null;
            }

            String prefix = matcher.group(1);
            int splitNumber = Integer.parseInt(matcher.group(2));
            int totalSplits = Integer.parseInt(matcher.group(3));

            return new SplitInfo(splitNumber, totalSplits, prefix);
        } catch (Exception e) {
            logger.error("Failed to parse split numbers from path: '{}', error: {}", splitPath, e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to parse split numbers from path: '%s', error: %s", splitPath, e.getMessage()), e);
        }
    }

    private static int getSystemMaxPath() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            return 32000; // Safety for Windows
        } else if (os.contains("mac")) {
            return 1024;
        } else {
            return 4096; // Linux/Unix
        }
    }

}