# Minecraft Vulkan Renderer
Minecraft Vulkan Renderer (MCVR) is a C++ implemented render framework for minecraft. It should be used with the Radiance mod.



# Build

First, make sure submodules are cloned with git.

```
git submodule update --init --recursive
```

## Linux

Use `cmake` to configure the project.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJAVA_PROJECT_ROOT_DIR=${PATH_TO_RADIANCE_JAVA_PROJECT} -DUSE_AMD=ON -DMCVR_ENABLE_NRD=ON
```

Build and install.

```
cmake --build build -j$(nproc)
cmake --install build
```

## Windows

Use `cmake` to configure the project.

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DJAVA_PROJECT_ROOT_DIR=${PATH_TO_RADIANCE_JAVA_PROJECT} -DMCVR_ENABLE_NRD=ON -DUSE_AMD=ON
```

Build and install.

```
cmake --build build --config Release
cmake --install build --config Release
```

