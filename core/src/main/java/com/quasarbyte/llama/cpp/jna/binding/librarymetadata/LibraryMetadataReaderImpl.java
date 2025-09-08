package com.quasarbyte.llama.cpp.jna.binding.librarymetadata;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.sun.jna.Library;
import com.sun.jna.NativeLibrary;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

public class LibraryMetadataReaderImpl implements LibraryMetadataReader {

    @Override
    public Path getLibraryPath(Library proxy) {
        Objects.requireNonNull(proxy);
        if (Proxy.isProxyClass(proxy.getClass())) {
            InvocationHandler ih = Proxy.getInvocationHandler(proxy);
            if (ih instanceof Library.Handler) {

                Library.Handler handler = (Library.Handler) ih;

                NativeLibrary nativeLib = handler.getNativeLibrary();

                return Paths.get(nativeLib.getFile().getAbsolutePath());
            }
        }

        throw new LlamaCppJnaException("Cannot get library path");
    }

}
