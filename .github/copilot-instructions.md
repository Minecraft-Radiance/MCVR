# MCVR — Project Guidelines

C++23 Vulkan rendering engine for Minecraft, used as a native backend for the [Radiance](../Radiance) Fabric mod. Outputs a shared library (`core.dll`/`libcore.so`) + compiled SPIR-V shaders consumed via JNI.

## Architecture

| Component | Path | Purpose |
|-----------|------|---------|
| JNI middleware | `src/core/middleware/` | Java↔C++ bridge (`RendererProxy`, `BufferProxy`, `TextureProxy`, `ChunkProxy`, `EntityProxy`) |
| Vulkan abstraction | `src/core/vulkan/` | Low-level Vulkan wrappers (Instance, Device, Swapchain, Pipeline, AS, SBT, etc.) |
| Render framework | `src/core/render/` | High-level rendering: `Renderer` singleton, `Framework`, `Pipeline`, `Chunks`, `Entities`, `Textures`, `Buffers` |
| Render modules | `src/core/render/modules/` | Pluggable pipeline stages: `ray_tracing`, `dlss`, `fsr_upscaler`, `xess_upscaler`, `nrd`, `svgf`, `temporal_accumulation`, `tone_mapping`, `post_render`, `overlay` |
| Shaders | `src/shader/` | GLSL → SPIR-V (Vulkan 1.4 target via `glslangValidator`), includes ray tracing shaders |
| Common | `src/common/` | Shared utilities: `singleton.hpp`, `shared.hpp` (ref-counted objects), `mapping.hpp` |
| External deps | `extern/` | Git submodules: glfw, volk, vma, glm, vulkan_headers, stb, DLSS, nrd, FidelityFX-SDK, sharc, minizip-ng, json, xess |

### Key patterns
- **SharedObject<T>**: Ref-counted base for Vulkan resource wrappers
- **Singleton**: `Renderer` is the global entry point
- **Module system**: Each render module registers via `Pipeline::worldModuleConstructors` static map with input/output image formats

## Build and Test

### Prerequisites
- CMake 3.15+, Visual Studio 2026 (or 2022), Vulkan SDK (with `glslangValidator` + `shaderc`), JDK 21 (for JNI headers)
- Radiance repo at `../Radiance` with JNI headers generated (`.\gradlew.bat compileJava`)

### Configure + Build
```powershell
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DJAVA_PROJECT_ROOT_DIR=C:/Users/gabri/Documents/CodingRepos/Radiance `
  -DMCVR_ENABLE_NRD=ON -DMCVR_ENABLE_FFX_UPSCALER=ON -DMCVR_ENABLE_XESS=ON -DUSE_AMD=OFF
cmake --build build --config Release
cmake --install build --config Release
```

### CMake options
| Option | Default | Effect |
|--------|---------|--------|
| `JAVA_PROJECT_ROOT_DIR` | **required** | Path to Radiance repo (for JNI headers + install destination) |
| `MCVR_ENABLE_NRD` | ON | NVIDIA Ray Tracing Denoiser |
| `MCVR_ENABLE_FFX_UPSCALER` | ON | AMD FidelityFX FSR3 upscaler |
| `MCVR_ENABLE_XESS` | ON | Intel XeSS upscaler |
| `USE_AMD` | ON | AMD-specific optimizations (set OFF for NVIDIA) |

### Install output
- `core.dll` / `core.lib` → `Radiance/src/main/resources/`
- `shaders/**/*.spv` → `Radiance/src/main/resources/shaders/`
- `internal.zip` (ray tracing GLSL sources) → `Radiance/src/main/resources/shaders/world/ray_tracing/`

## Conventions

- C++23 standard, MSVC on Windows
- `VK_NO_PROTOTYPES` defined — Vulkan loaded dynamically via volk
- GLFW function pointers rebound from Java's LWJGL DLL at runtime (no static GLFW linking)
- Shader compilation: GLSL → SPIR-V via `glslangValidator --target-env vulkan1.4`
- NRD shaders compiled via ShaderMake (downloaded during CMake configure)
- `NOMINMAX` and `WIN32_LEAN_AND_MEAN` defined on MSVC

## Pitfalls

- First CMake configure downloads DLSS libs, ShaderMake, DXC, and NRD dependencies — requires internet and can be slow
- `JAVA_PROJECT_ROOT_DIR` must use forward slashes even on Windows (CMake normalizes it)
- JNI headers must be regenerated (`gradlew compileJava` in Radiance) whenever native method signatures change
- The `glslangValidator` must be on PATH or found via `VULKAN_SDK` env var
