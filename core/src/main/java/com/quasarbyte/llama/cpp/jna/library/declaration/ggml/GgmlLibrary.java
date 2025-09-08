package com.quasarbyte.llama.cpp.jna.library.declaration.ggml;

import com.quasarbyte.llama.cpp.jna.library.declaration.LlamaCppBase;
import com.sun.jna.Pointer;

// Separate interface for GGML backend functions with explicit calling convention
public interface GgmlLibrary extends LlamaCppBase<GgmlLibrary> {
    // GGML backend loading functions

    /**
     * Loads all registered GGML backends using the default loader search paths.
     * <p>
     * Under the hood this simply calls {@code ggml_backend_load_all_from_path(null)} which
     * enumerates a narrow set of locations: the compile-time {@code GGML_BACKEND_DIR} (if
     * defined), the directory that hosts the running executable, and the current working
     * directory ({@code ggml-backend-reg.cpp:506-513}). When running from Java these
     * usually resolve to the JDK installation and the project root, so no plugins are
     * found unless the backend DLLs are copied there. The loader also checks the optional
     * {@code GGML_BACKEND_PATH} environment variable, but in this code path it expects a
     * concrete library file rather than a directory ({@code ggml-backend-reg.cpp:600}).
     * Because of these constraints this method often returns without loading anything in
     * typical JVM setups.
     */
    void ggml_backend_load_all();

    /**
     * Loads a specific GGML backend from the supplied path.
     * <p>
     * The path must point directly to the backend shared library (for example
     * {@code .../ggml-cpu.dll}). Passing a directory will fail silently because the loader
     * ultimately calls {@code LoadLibrary} on the string.
     *</p>
     *
     * @param path absolute or relative path to a backend library file
     * @return native backend registry handle ({@code ggml_backend_reg_t})
     */
    Pointer ggml_backend_load(String path); // Returns ggml_backend_reg_t

    /**
     * Loads backends by scanning the provided directory for {@code ggml-<name>-*.dll}
     * (or platform equivalents).
     * <p>
     * This is the recommended entry point when running from Java because it bypasses the
     * default search limitations of {@link #ggml_backend_load_all()}. The directory should
     * point to the folder that contains the GGML backend binaries you wish to load (for
     * example the value returned by {@code GGML_BACKEND_PATH_ENV_VAR_NAME} in
     * {@code SimpleChatOriginal.java}).
     *</p>
     * <p>
     * The underlying llama.cpp loader still resolves each shared library through the
     * operating system search path. Make sure the directory that contains the native
     * GGML / llama binaries is on the relevant path variable for your platform (for
     * example add it to {@code PATH} on Windows, {@code LD_LIBRARY_PATH} on Linux, or
     * {@code DYLD_LIBRARY_PATH} on macOS). Providing {@code ggml.backend.path} or
     * {@code GGML_BACKEND_PATH} merely tells the loader where to look for plugin files;
     * it does not bypass the dynamic loader or relax the PATH requirement. If those
     * paths are missing the libraries will fail to load.
     *</p>
     * <p>
     * Windows users must also match the Microsoft Visual C++ runtime that ships with the
     * {@code llama.cpp} build. Modern releases (for example build {@code b6527}) require the
     * redistributable packaged with the latest Visual Studio toolset. JDK 8 through JDK 24 on
     * Windows bundle older copies of {@code vcruntime140.dll}, {@code vcruntime140_1.dll}, and
     * {@code msvcp140.dll} inside {@code <java.home>/bin}. The JVM loads those copies before it
     * probes the system search path, which prevents the newer runtime from being located and results
     * in {@code UnsatisfiedLinkError} or Windows error {@code 0xc0000135} when the llama bindings
     * initialise. Pick one of the following strategies when you cannot use a compatible runtime out
     * of the box:
     * <ul>
     *     <li>Prefer JDK 25 and newer on Windows; these distributions ship DLLs that work
     *         with recent {@code llama.cpp} builds without changes.</li>
     *     <li>If you must stay on JDK 8–24, temporarily move, rename, or archive the bundled
     *         {@code vcruntime140*.dll} and {@code msvcp140.dll} files so that Windows falls back to
     *         the Microsoft Visual C++ Redistributable you installed separately.</li>
     *     <li>Advanced users can launch the JVM through a custom bootstrapper that amends
     *         {@code PATH} (or uses {@code SetDllDirectory}) before any native library loads.</li>
     * </ul>
     * When the bundled runtime wins the race you may see crashes such as:
     * <pre>
     * Exception in thread "main" java.lang.Error: Invalid memory access
     *     at com.sun.jna.Native.invokeVoid(Native Method)
     *     at com.sun.jna.Function.invoke(Function.java:418)
     *     at com.sun.jna.Function.invoke(Function.java:364)
     *     at com.sun.jna.Library$Handler.invoke(Library.java:270)
     *     at com.sun.proxy.$Proxy0.ggml_backend_load_all_from_path(Unknown Source)
     *     at com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader.GgmlBackendLoaderImpl.loadBackend(GgmlBackendLoaderImpl.java:29)
     * </pre>
     * WinDbg traces show the failure originates in {@code llama.cpp/ggml/src/ggml-backend-reg.cpp}
     * inside {@code static dl_handle * dl_load_library(const fs::path & path)} where the call
     * {@code HMODULE handle = LoadLibraryW(path.wstring().c_str());} returns {@code NULL} because the
     * process already loaded incompatible copies of {@code vcruntime140.dll} / {@code msvcp140.dll}.
     * The debugger captures a typical load order that triggers the fault:
     * <pre>
     * llama.dll
     * ├── ggml-cuda.dll
     * │   ├── cudart64_**.dll
     * │   │   ├── cudart64_12.dll
     * │   │   ├── nvcuda.dll
     * │   │   ├── cublas64_12.dll
     * │   │   └── cublasLt64_12.dll
     * │   ├── vcruntime140.dll   (from JDK bin)
     * │   └── msvcp140.dll       (from JDK bin)
     * └── ...
     * </pre>
     * WinDbg module logs mirror the same sequence:
     * <pre>
     * ModLoad: ggml.dll
     * ModLoad: ggml-base.dll
     * ModLoad: ggml-cuda.dll   &lt;-- crash on first mutex initialisation
     * ModLoad: cudart64_12.dll
     * ModLoad: nvcuda.dll
     * ModLoad: cublas64_12.dll
     * ModLoad: cublasLt64_12.dll
     * </pre>
     * On CUDA-enabled builds the crash frequently surfaces while the backend iterates over GPU
     * plugins: the MSVC runtime resolves mutex primitives in {@code mtx_do_lock} against the wrong
     * ABI, so the first CUDA DLL that touches a mutex dereferences a null pointer and the JVM sees an
     * access violation. These errors disappear once Windows resolves the DLLs from the Visual C++
     * Redistributable that
     * matches your {@code llama.cpp} build.
     * Always ensure the matching Microsoft Visual C++ Redistributable is present on the machine
     * before removing DLLs from the JDK. Without a compatible runtime the {@code llama.cpp} native
     * libraries cannot be loaded.
     *</p>
     *
     * @param path directory containing backend libraries; {@code null} mirrors the default
     *             search behaviour
     */
    void ggml_backend_load_all_from_path(String path);

    Pointer ggml_backend_unload(Pointer backend_reg);

    // Backend initialization functions
    Pointer ggml_backend_init_by_name(String name);

    Pointer ggml_backend_init_by_type(int type);

    Pointer ggml_backend_init_best();

    // Backend registry functions
    int ggml_backend_reg_count();

    Pointer ggml_backend_reg_get(int index);

    Pointer ggml_backend_reg_by_name(String name);

    void ggml_backend_register(Pointer backend_reg);

    void ggml_backend_device_register(Pointer device_reg);

    // Device management functions
    int ggml_backend_dev_count();

    Pointer ggml_backend_dev_get(int index);

    Pointer ggml_backend_dev_by_name(String name);

    Pointer ggml_backend_dev_by_type(int type);
}
