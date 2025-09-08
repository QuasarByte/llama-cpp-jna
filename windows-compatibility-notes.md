# Windows Compatibility Notes

The prebuilt Windows binaries for `llama.cpp` (including build `b6527`) are linked against the
latest Microsoft Visual C++ Redistributable. When you launch the bindings through the JVM, the
Java distribution may bring its own copy of the MSVC runtime:

- JDK 25+ ship DLLs that are compatible with recent `llama.cpp` builds and work without
  any changes.
- JDK 8–24 bundle older versions of `vcruntime140.dll`, `vcruntime140_1.dll`, and `msvcp140.dll`
  under `<java.home>/bin`. The JVM loads those DLLs first, which prevents Windows from resolving the
  newer runtime and causes native loading errors.

If you are running on JDK 8–24 for legacy reasons, either migrate to JDK 25+, or remove/rename the
bundled MSVC runtime DLLs so that Windows instead loads the redistributable you installed globally.
Ensure the matching Microsoft Visual C++ Redistributable is present on the system before you remove
files from the JDK. Advanced setups can also launch the JVM with a custom bootstrapper that adjusts
`PATH` (or calls `SetDllDirectory`) before any native code is loaded. When the wrong runtime wins the
race, the failure surfaces inside `llama.cpp/ggml/src/ggml-backend-reg.cpp` during the call to
`LoadLibraryW` in `dl_load_library(...)`; WinDbg confirms `LoadLibraryW` returns `NULL` once
`vcruntime140.dll`/`msvcp140.dll` from the JDK are already mapped. Typical failing loader chains look
like this:

```
llama.dll
├── ggml-cuda.dll
│   ├── cudart64_**.dll
│   │   ├── cudart64_12.dll
│   │   ├── nvcuda.dll
│   │   ├── cublas64_12.dll
│   │   └── cublasLt64_12.dll
│   ├── vcruntime140.dll   (from JDK bin)
│   └── msvcp140.dll       (from JDK bin)
└── ...
```

WinDbg module logs surface the same order:

```
ModLoad: ggml.dll
ModLoad: ggml-base.dll
ModLoad: ggml-cuda.dll   <-- crash when the first mutex is initialised
ModLoad: cudart64_12.dll
ModLoad: nvcuda.dll
ModLoad: cublas64_12.dll
ModLoad: cublasLt64_12.dll
```

CUDA-enabled builds make the
problem easier to reproduce because the first GPU plugin that acquires a C++ mutex trips over
`msvcp140!mtx_do_lock` with a null control block, yielding an access violation.

## Helpfully links
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170
