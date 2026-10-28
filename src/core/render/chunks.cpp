#include "core/render/chunks.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/vertex.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

#ifdef _WIN32
#    include <windows.h>
#elif defined(__linux__)
#    include <sys/resource.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

struct LightData {
    glm::vec4 p0Area;
    glm::vec4 p1;
    glm::vec4 p2;
    glm::vec4 p3;
    glm::vec4 normal;
    glm::vec4 radiance;
    glm::vec4 sourceIDData;
};
static_assert(sizeof(LightData) == sizeof(glm::vec4) * 7);

static uint64_t hashCombine64(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

static VkDeviceSize alignDeviceSize(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) { return value; }
    return ((value + alignment - 1) / alignment) * alignment;
}

static VkDeviceSize growDeviceBufferCapacity(VkDeviceSize current, VkDeviceSize required, VkDeviceSize alignment) {
    (void)current;
    const VkDeviceSize alignedRequired = alignDeviceSize(required, alignment);
    const VkDeviceSize slack = std::min<VkDeviceSize>(alignedRequired / 16, 64ull * 1024ull);
    VkDeviceSize target = alignedRequired + slack;
    return alignDeviceSize(target, alignment);
}

static constexpr size_t chunkGeometryStagingCacheBudgetBytes = 256ull * 1024ull * 1024ull;

static std::shared_ptr<vk::HostVisibleStagingBufferPool> chunkGeometryStagingPool(const std::shared_ptr<vk::VMA> &vma) {
    static std::mutex mutex;
    static std::weak_ptr<vk::HostVisibleStagingBufferPool> weakPool;

    std::lock_guard<std::mutex> lock(mutex);
    auto pool = weakPool.lock();
    if (pool == nullptr) {
        pool = vk::HostVisibleStagingBufferPool::create(vma, chunkGeometryStagingCacheBudgetBytes,
                                                        vk::ResourceMemoryCategory::Chunk);
        weakPool = pool;
    }
    MCVR_PROFILE_METRIC("chunk_upload.staging_cache_budget_bytes", chunkGeometryStagingCacheBudgetBytes);
    MCVR_PROFILE_METRIC("chunk_upload.staging_cache_cached_bytes", pool->cachedBytes());
    MCVR_PROFILE_METRIC("chunk_upload.staging_cache_cached_count", pool->cachedCount());
    return pool;
}

std::shared_ptr<vk::DeviceLocalBuffer> ChunkGpuBufferPool::acquire(const std::shared_ptr<vk::VMA> &vma,
                                                                   const std::shared_ptr<vk::Device> &device,
                                                                   uint64_t frameSerial,
                                                                   VkDeviceSize requiredBytes,
                                                                   VkBufferUsageFlags usage,
                                                                   bool persistStaging,
                                                                   VmaMemoryUsage memoryUsage,
                                                                   VkDeviceSize alignment) {
    MCVR_PROFILE_SCOPE("native.ChunkGpuBufferPool.acquire");
    MCVR_PROFILE_METRIC("pool.acquire_required_bytes", requiredBytes);
    if (requiredBytes == 0) { return nullptr; }
    auto stagingPool = persistStaging ? chunkGeometryStagingPool(vma) : nullptr;

    std::unique_lock<std::mutex> lock(mutex_);
    MCVR_PROFILE_METRIC("pool.acquire_entries", entries_.size());
    size_t bestIndex = entries_.size();
    VkDeviceSize largestCapacity = 0;
    size_t reusableEntryCount = 0;
    for (size_t i = 0; i < entries_.size(); i++) {
        const auto &entry = entries_[i];
        if (entry.buffer == nullptr) { continue; }
        largestCapacity = std::max<VkDeviceSize>(largestCapacity, entry.buffer->size());
        if (entry.reusableFrameSerial > frameSerial || entry.buffer->size() < requiredBytes) { continue; }
        reusableEntryCount++;
        if (bestIndex == entries_.size() || entry.buffer->size() < entries_[bestIndex].buffer->size()) {
            bestIndex = i;
        }
    }
    MCVR_PROFILE_METRIC("pool.acquire_reusable_entries", reusableEntryCount);

    if (bestIndex != entries_.size()) {
        auto buffer = entries_[bestIndex].buffer;
        MCVR_PROFILE_METRIC("pool.acquire_hit", 1);
        MCVR_PROFILE_METRIC("pool.acquire_capacity_bytes", buffer->size());
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(bestIndex));
        entryCount_.store(entries_.size(), std::memory_order_release);
        buffer->setStagingBufferPool(stagingPool);
        return buffer;
    }

    const VkDeviceSize capacity = growDeviceBufferCapacity(largestCapacity, requiredBytes, alignment);
    MCVR_PROFILE_METRIC("pool.acquire_hit", 0);
    MCVR_PROFILE_METRIC("pool.acquire_capacity_bytes", capacity);
    lock.unlock();
    auto buffer =
        vk::DeviceLocalBuffer::create(vma, device, persistStaging, capacity, usage, 0, memoryUsage, alignment);
    buffer->setMemoryCategory(vk::ResourceMemoryCategory::Chunk);
    buffer->setStagingBufferPool(stagingPool);
    return buffer;
}

void ChunkGpuBufferPool::release(std::shared_ptr<vk::DeviceLocalBuffer> buffer, uint64_t reusableFrameSerial) {
    MCVR_PROFILE_SCOPE("native.ChunkGpuBufferPool.release");
    if (buffer == nullptr) { return; }
    MCVR_PROFILE_METRIC("pool.release_capacity_bytes", buffer->size());
    std::unique_lock<std::mutex> lock(mutex_);
    entries_.push_back({
        .buffer = std::move(buffer),
        .reusableFrameSerial = reusableFrameSerial,
    });
    entryCount_.store(entries_.size(), std::memory_order_release);
    MCVR_PROFILE_METRIC("pool.release_entries", entries_.size());
}

void ChunkGpuBufferPool::sweep(uint64_t frameSerial, bool aggressive) {
    if (entryCount_.load(std::memory_order_acquire) == 0) { return; }
    static constexpr uint64_t maxIdleFrames = 120;
    std::unique_lock<std::mutex> lock(mutex_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [frameSerial, aggressive](const Entry &entry) {
                                      if (entry.buffer == nullptr) { return true; }
                                      if (entry.reusableFrameSerial > frameSerial) { return false; }
                                      return aggressive || frameSerial - entry.reusableFrameSerial > maxIdleFrames;
                                  }),
                   entries_.end());
    entryCount_.store(entries_.size(), std::memory_order_release);
}

void ChunkGpuBufferPool::clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    entries_.clear();
    entryCount_.store(0, std::memory_order_release);
    sweepQueued_.store(false, std::memory_order_release);
}

bool ChunkGpuBufferPool::hasEntries() const {
    return entryCount_.load(std::memory_order_acquire) != 0;
}

bool ChunkGpuBufferPool::markSweepQueued() {
    bool expected = false;
    return sweepQueued_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
}

void ChunkGpuBufferPool::clearSweepQueued() {
    sweepQueued_.store(false, std::memory_order_release);
}

ChunkGpuBufferPoolStats ChunkGpuBufferPool::stats() const {
    std::unique_lock<std::mutex> lock(mutex_);
    ChunkGpuBufferPoolStats result{
        .entryCount = entries_.size(),
    };
    for (const auto &entry : entries_) {
        if (entry.buffer != nullptr) { result.capacityBytes += entry.buffer->size(); }
    }
    return result;
}

static float packUintBits(uint32_t value) {
    float packed = 0.0f;
    std::memcpy(&packed, &value, sizeof(uint32_t));
    return packed;
}

static float triangleArea(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c) {
    return 0.5f * glm::length(glm::cross(b - a, c - a));
}

static float cross2D(const glm::vec2 &a, const glm::vec2 &b) {
    return a.x * b.y - a.y * b.x;
}

static void setCurrentChunkBuildThreadLowPriority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
#endif
}

static void clipUvPolygonAgainstEdge(
    const std::vector<glm::vec2> &input, std::vector<glm::vec2> &output, int axis, float value, bool keepGreater) {
    output.clear();
    if (input.empty()) { return; }

    auto coordinate = [&](const glm::vec2 &uv) -> float { return axis == 0 ? uv.x : uv.y; };
    auto inside = [&](const glm::vec2 &uv) -> bool {
        return keepGreater ? coordinate(uv) >= value : coordinate(uv) <= value;
    };
    auto intersect = [&](const glm::vec2 &a, const glm::vec2 &b) -> glm::vec2 {
        float delta = coordinate(b) - coordinate(a);
        if (std::abs(delta) <= 1e-8f) { return a; }
        float t = (value - coordinate(a)) / delta;
        return glm::mix(a, b, std::clamp(t, 0.0f, 1.0f));
    };

    glm::vec2 previous = input.back();
    bool previousInside = inside(previous);
    for (const glm::vec2 &current : input) {
        bool currentInside = inside(current);
        if (currentInside != previousInside) { output.push_back(intersect(previous, current)); }
        if (currentInside) { output.push_back(current); }
        previous = current;
        previousInside = currentInside;
    }
}

static std::vector<glm::vec2>
clipUvTriangleToRect(const std::array<glm::vec2, 3> &triangle, const glm::vec2 &rectMin, const glm::vec2 &rectMax) {
    std::vector<glm::vec2> polygon(triangle.begin(), triangle.end());
    std::vector<glm::vec2> scratch;

    clipUvPolygonAgainstEdge(polygon, scratch, 0, rectMin.x, true);
    polygon.swap(scratch);
    clipUvPolygonAgainstEdge(polygon, scratch, 0, rectMax.x, false);
    polygon.swap(scratch);
    clipUvPolygonAgainstEdge(polygon, scratch, 1, rectMin.y, true);
    polygon.swap(scratch);
    clipUvPolygonAgainstEdge(polygon, scratch, 1, rectMax.y, false);
    polygon.swap(scratch);
    return polygon;
}

static void releaseChunkBuildDataNonPooledStaging(const std::shared_ptr<ChunkBuildData> &chunkBuildData) {
    if (chunkBuildData->lightBuffer != nullptr) { chunkBuildData->lightBuffer->releaseStaging(); }
}

static void releaseChunkBuildDataGeometryStaging(const std::shared_ptr<ChunkBuildData> &chunkBuildData) {
    MCVR_PROFILE_SCOPE("native.releaseChunkBuildDataGeometryStaging");
    if (chunkBuildData == nullptr) { return; }

    VkDeviceSize releasedBytes = 0;
    uint32_t releasedCount = 0;
    auto releaseGeometryStaging = [&releasedBytes,
                                   &releasedCount](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer) {
        if (buffer == nullptr || !buffer->hasStagingBuffer()) { return; }
        releasedBytes += buffer->size();
        releasedCount++;
        buffer->releaseStagingDeferred();
    };

    releaseGeometryStaging(chunkBuildData->indexBuffer);
    releaseGeometryStaging(chunkBuildData->pbrBuffer);
    MCVR_PROFILE_METRIC("chunk_upload.release_geometry_staging_bytes", releasedBytes);
    MCVR_PROFILE_METRIC("chunk_upload.release_geometry_staging_count", releasedCount);
}

static void releaseChunkBuildDataInputs(const std::shared_ptr<ChunkBuildData> &chunkBuildData) {
    if (chunkBuildData == nullptr) { return; }
    chunkBuildData->releaseBuildInputs();
}

static uint64_t reusableFrameSerialFromNow() {
    auto world = Renderer::instance().world();
    auto chunks = world != nullptr ? world->chunks() : nullptr;
    if (chunks == nullptr) { return 0; }
    return chunks->frameSerial() + chunks->bufferReuseDelayFrames();
}

static void releaseBufferToPool(const std::shared_ptr<ChunkGpuBufferPool> &pool,
                                std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                uint64_t reusableFrameSerial) {
    MCVR_PROFILE_SCOPE("native.releaseBufferToPool");
    if (buffer == nullptr) { return; }
    MCVR_PROFILE_METRIC("pool.release_buffer_bytes", buffer->size());
    if (buffer->hasStagingBuffer()) {
        MCVR_PROFILE_METRIC("chunk_upload.release_pool_staging_bytes", buffer->size());
        buffer->releaseStagingDeferred();
    }
    if (pool == nullptr) { throw std::runtime_error("chunk GPU buffer pool is not initialized"); }
    pool->release(buffer, reusableFrameSerial);
    buffer = nullptr;
    auto world = Renderer::instance().world();
    auto chunks = world != nullptr ? world->chunks() : nullptr;
    if (chunks != nullptr) { chunks->trackChunkBufferPoolForSweep(pool); }
}

static void releaseChunkBuildDataBuffersToPools(const std::shared_ptr<ChunkBuildData> &chunkBuildData,
                                                bool releaseActiveBuffers) {
    MCVR_PROFILE_SCOPE("native.releaseChunkBuildDataBuffersToPools");
    MCVR_PROFILE_METRIC("pool.release_active_buffers", releaseActiveBuffers ? 1 : 0);
    if (chunkBuildData == nullptr) { return; }

    const uint64_t reusableFrameSerial = reusableFrameSerialFromNow();
    if (releaseActiveBuffers) {
        releaseBufferToPool(chunkBuildData->indexBufferPool, chunkBuildData->indexBuffer, reusableFrameSerial);
        releaseBufferToPool(chunkBuildData->pbrBufferPool, chunkBuildData->pbrBuffer, reusableFrameSerial);
        releaseBufferToPool(chunkBuildData->blasStoragePool, chunkBuildData->blasStorageBuffer, reusableFrameSerial);
    }
    releaseBufferToPool(chunkBuildData->blasScratchPool, chunkBuildData->blasScratchBuffer, reusableFrameSerial);
}

static void bindChunkBuildDataPools(const std::shared_ptr<ChunkBuildData> &chunkBuildData,
                                    const std::shared_ptr<Chunk1> &chunk) {
    if (chunkBuildData == nullptr || chunk == nullptr) { return; }
    chunkBuildData->indexBufferPool = chunk->indexBufferPool;
    chunkBuildData->pbrBufferPool = chunk->pbrBufferPool;
    chunkBuildData->blasStoragePool = chunk->blasStoragePool;
    chunkBuildData->blasScratchPool = chunk->blasScratchPool;
}

static constexpr uint32_t unusedChunkSlot = std::numeric_limits<uint32_t>::max();

struct RenderableChunkSlotUpdateInfo {
    bool hasChanged = false;
    bool isRenderable = false;
    bool wasRemoved = false;
    uint32_t renderSlot = unusedChunkSlot;
    int64_t movedChunkId = -1;
    uint32_t movedRenderSlot = unusedChunkSlot;
};

static RenderableChunkSlotUpdateInfo
updateRenderableChunkSlot(std::vector<std::shared_ptr<Chunk1>> &renderableChunks,
                          std::vector<std::shared_ptr<ChunkRenderData>> &renderableChunkSnapshots,
                          std::vector<int64_t> &renderableChunkIds,
                          std::vector<uint32_t> &renderableChunkSlots,
                          int64_t chunkId,
                          const std::shared_ptr<Chunk1> &chunk) {
    RenderableChunkSlotUpdateInfo slotUpdateInfo{};
    const size_t chunkIndex = static_cast<size_t>(chunkId);
    const bool isRenderable = chunk != nullptr && chunk->blas != nullptr;
    uint32_t &slot = renderableChunkSlots[chunkIndex];

    if (isRenderable) {
        if (slot == unusedChunkSlot) {
            slot = static_cast<uint32_t>(renderableChunks.size());
            renderableChunks.push_back(chunk);
            renderableChunkSnapshots.push_back(chunk->renderDataSnapshot());
            renderableChunkIds.push_back(chunkId);
        } else {
            renderableChunks[slot] = chunk;
            renderableChunkSnapshots[slot] = chunk->renderDataSnapshot();
        }
        slotUpdateInfo.hasChanged = true;
        slotUpdateInfo.isRenderable = true;
        slotUpdateInfo.renderSlot = slot;
        return slotUpdateInfo;
    }

    if (slot == unusedChunkSlot) { return slotUpdateInfo; }

    const uint32_t removedSlot = slot;
    const uint32_t lastSlot = static_cast<uint32_t>(renderableChunks.size() - 1);
    if (removedSlot != lastSlot) {
        renderableChunks[removedSlot] = std::move(renderableChunks[lastSlot]);
        renderableChunkSnapshots[removedSlot] = std::move(renderableChunkSnapshots[lastSlot]);
        renderableChunkIds[removedSlot] = renderableChunkIds[lastSlot];
        renderableChunkSlots[static_cast<size_t>(renderableChunkIds[removedSlot])] = removedSlot;
        slotUpdateInfo.movedChunkId = renderableChunkIds[removedSlot];
        slotUpdateInfo.movedRenderSlot = removedSlot;
    }
    renderableChunks.pop_back();
    renderableChunkSnapshots.pop_back();
    renderableChunkIds.pop_back();
    slot = unusedChunkSlot;
    slotUpdateInfo.hasChanged = true;
    slotUpdateInfo.wasRemoved = true;
    slotUpdateInfo.renderSlot = removedSlot;
    return slotUpdateInfo;
}

static void applyRenderableChunkSlotUpdate(ChunkGeometryMetadataCache &chunkMetadataCache,
                                           const std::vector<std::shared_ptr<Chunk1>> &chunks,
                                           const RenderableChunkSlotUpdateInfo &slotUpdateInfo,
                                           int64_t chunkId,
                                           size_t renderableChunkCount) {
    if (!slotUpdateInfo.hasChanged) { return; }
    if (slotUpdateInfo.wasRemoved) { chunkMetadataCache.removeChunk(static_cast<uint32_t>(chunkId)); }
    if (slotUpdateInfo.movedChunkId >= 0) {
        chunkMetadataCache.updateRenderRecord(static_cast<uint32_t>(slotUpdateInfo.movedChunkId),
                                              *chunks[slotUpdateInfo.movedChunkId], slotUpdateInfo.movedRenderSlot);
    }
    if (slotUpdateInfo.isRenderable) {
        chunkMetadataCache.updateChunk(static_cast<uint32_t>(chunkId), *chunks[chunkId], slotUpdateInfo.renderSlot);
    }
    chunkMetadataCache.resizeRenderRecords(renderableChunkCount);
}

static bool requiresGpuSubmit(const std::shared_ptr<ChunkBuildDataBatch> &chunkBuildDataBatch) {
    if (!chunkBuildDataBatch->blasBuilders.empty()) { return true; }
    for (const auto &chunkBuildData : chunkBuildDataBatch->batchData) {
        if (chunkBuildData != nullptr && chunkBuildData->lightBuffer != nullptr) { return true; }
    }
    return false;
}

static bool
barycentricFromUv(const glm::vec2 &uv, const std::array<glm::vec2, 3> &triangleUv, glm::vec3 &outBarycentric) {
    glm::vec2 e0 = triangleUv[1] - triangleUv[0];
    glm::vec2 e1 = triangleUv[2] - triangleUv[0];
    float denominator = cross2D(e0, e1);
    if (std::abs(denominator) <= 1e-8f) { return false; }

    glm::vec2 relative = uv - triangleUv[0];
    float b1 = cross2D(relative, e1) / denominator;
    float b2 = cross2D(e0, relative) / denominator;
    outBarycentric = glm::vec3(1.0f - b1 - b2, b1, b2);
    return true;
}

static bool uvToTrianglePosition(const glm::vec2 &uv,
                                 const std::array<glm::vec2, 3> &triangleUv,
                                 const std::array<glm::vec3, 3> &trianglePos,
                                 glm::vec3 &outPosition) {
    glm::vec3 barycentric;
    if (!barycentricFromUv(uv, triangleUv, barycentric)) { return false; }

    outPosition = trianglePos[0] * barycentric.x + trianglePos[1] * barycentric.y + trianglePos[2] * barycentric.z;
    return true;
}

static bool uvToTriangleTint(const glm::vec2 &uv,
                             const std::array<glm::vec2, 3> &triangleUv,
                             const std::array<glm::vec3, 3> &triangleTint,
                             glm::vec3 &outTint) {
    glm::vec3 barycentric;
    if (!barycentricFromUv(uv, triangleUv, barycentric)) { return false; }

    outTint = triangleTint[0] * barycentric.x + triangleTint[1] * barycentric.y + triangleTint[2] * barycentric.z;
    return true;
}

static glm::vec3 emissionVertexTint(const vk::VertexFormat::ChunkPBRVertex &vertex) {
    if (!vk::Vertex::hasFlag(vertex.packedData, vk::Vertex::useColorLayerBit)) { return glm::vec3(1.0f); }

    return glm::clamp(glm::vec3(vk::Vertex::unpackColorRGBA8(vertex.colorRGBA8)), glm::vec3(0.0f), glm::vec3(1.0f));
}

static uint64_t
buildChunkLightStableID(int64_t chunkId, uint32_t geometryIndex, uint32_t quadIndex, uint64_t cellStableKey) {
    uint64_t seed = 0xcbf29ce484222325ull;
    seed = hashCombine64(seed, static_cast<uint64_t>(chunkId));
    seed = hashCombine64(seed, static_cast<uint64_t>(geometryIndex));
    seed = hashCombine64(seed, static_cast<uint64_t>(quadIndex));
    seed = hashCombine64(seed, cellStableKey);
    return seed;
}

static ChunkPackedData makeChunkPackedData(
    int32_t x, int32_t y, int32_t z, uint32_t geometryCount, uint32_t lightCount, VkDeviceAddress lightBufferAddress) {
    return {
        .x = x,
        .y = y,
        .z = z,
        .geometryCount = geometryCount,
        .lightCount = lightCount,
        .lightBufferAddress = lightBufferAddress,
        .entityLightCount = 0,
        .entityLightBufferAddress = 0,
        .globalEntityLightCount = 0,
        .globalEntityLightBufferAddress = 0,
    };
}

static void storeChunkPackedData(std::vector<ChunkPackedData> &chunkPackedData,
                                 int64_t id,
                                 int32_t x,
                                 int32_t y,
                                 int32_t z,
                                 uint32_t geometryCount,
                                 uint32_t lightCount,
                                 VkDeviceAddress lightBufferAddress) {
    if (!Renderer::options.collectChunkEmission) { return; }

    ChunkPackedData &data = chunkPackedData[id];
    data.x = x;
    data.y = y;
    data.z = z;
    data.geometryCount = geometryCount;
    data.lightCount = lightCount;
    data.lightBufferAddress = lightBufferAddress;
}

static glm::uvec4 packDeviceAddress(VkDeviceAddress address) {
    const uint64_t value = static_cast<uint64_t>(address);
    return glm::uvec4(static_cast<uint32_t>(value & 0xffffffffull),
                      static_cast<uint32_t>((value >> 32ull) & 0xffffffffull), 0, 0);
}

template <typename T>
static std::shared_ptr<vk::DeviceLocalBuffer> createMetadataBuffer(const std::vector<T> &data,
                                                                   const std::shared_ptr<vk::VMA> &vma,
                                                                   const std::shared_ptr<vk::Device> &device,
                                                                   VkBufferUsageFlags usage) {
    if (data.empty()) { return nullptr; }

    auto buffer = vk::DeviceLocalBuffer::create(vma, device, false, data.size() * sizeof(T),
                                                usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    buffer->setMemoryCategory(vk::ResourceMemoryCategory::Chunk);
    buffer->uploadToStagingBuffer(const_cast<T *>(data.data()), data.size() * sizeof(T), 0);
    return buffer;
}

static uint32_t calculateChunkGeometryCapacity(uint32_t geometryCount) {
    uint32_t capacity = 2;
    while (capacity < geometryCount) { capacity <<= 1; }
    return capacity;
}

static void mergeDirtyRange(std::vector<ChunkMetadataUpdateRangeInfo> &ranges, uint32_t offset, uint32_t count) {
    if (count == 0) { return; }

    uint32_t end = offset + count;
    auto first = std::lower_bound(ranges.begin(), ranges.end(), offset,
                                  [](const auto &range, uint32_t begin) { return range.offset + range.count < begin; });
    auto last = first;
    while (last != ranges.end() && last->offset <= end) {
        offset = std::min(offset, last->offset);
        end = std::max(end, last->offset + last->count);
        ++last;
    }
    if (first == last) {
        ranges.insert(first, {.offset = offset, .count = end - offset});
    } else {
        first->offset = offset;
        first->count = end - offset;
        ranges.erase(first + 1, last);
    }
}

static ChunkMetadataSlabInfo allocateChunkMetadataSlab(ChunkGeometryMetadataCache &cache, uint32_t geometryCapacity) {
    auto &freeSlabs = cache.freeSlabsByCapacity[geometryCapacity];
    if (!freeSlabs.empty()) {
        ChunkMetadataSlabInfo slab = freeSlabs.back();
        freeSlabs.pop_back();
        return slab;
    }

    ChunkMetadataSlabInfo slab{
        .geometryOffset = static_cast<uint32_t>(cache.indexBufferAddresses.size()),
        .hitGroupOffset = static_cast<uint32_t>(cache.hitGroupNames.size()),
    };
    cache.indexBufferAddresses.resize(static_cast<size_t>(slab.geometryOffset) + geometryCapacity);
    cache.pbrBufferAddresses.resize(static_cast<size_t>(slab.geometryOffset) + geometryCapacity);
    cache.hitGroupNames.resize(static_cast<size_t>(slab.hitGroupOffset) + geometryCapacity + 1, nullptr);
    cache.geometrySlotCount = cache.indexBufferAddresses.size();
    return slab;
}

static void clearChunkMetadataSlab(ChunkGeometryMetadataCache &cache,
                                   const ChunkMetadataSlabAllocationInfo &allocation) {
    if (!allocation.allocated()) { return; }

    std::fill(cache.hitGroupNames.begin() + allocation.hitGroupOffset,
              cache.hitGroupNames.begin() + allocation.hitGroupOffset + allocation.geometryCapacity + 1, nullptr);
    std::fill(cache.indexBufferAddresses.begin() + allocation.geometryOffset,
              cache.indexBufferAddresses.begin() + allocation.geometryOffset + allocation.geometryCapacity, 0);
    std::fill(cache.pbrBufferAddresses.begin() + allocation.geometryOffset,
              cache.pbrBufferAddresses.begin() + allocation.geometryOffset + allocation.geometryCapacity, 0);
    mergeDirtyRange(cache.hitGroupDirtyRanges, allocation.hitGroupOffset, allocation.geometryCapacity + 1);
    mergeDirtyRange(cache.geometryDirtyRanges, allocation.geometryOffset, allocation.geometryCapacity);
}

void ChunkGeometryMetadataCache::reset(uint32_t chunkCount) {
    clear();
    chunkAllocations.assign(chunkCount, ChunkMetadataSlabAllocationInfo{});
}

void ChunkGeometryMetadataCache::clear() {
    hitGroupNames.clear();
    indexBufferAddresses.clear();
    pbrBufferAddresses.clear();
    renderRecords.clear();
    chunkAllocations.clear();
    freeSlabsByCapacity.clear();
    renderRecordDirtyRanges.clear();
    geometryDirtyRanges.clear();
    hitGroupDirtyRanges.clear();
    rasterDrawDirtySlots.clear();
    rasterDrawDirtySlotMarks.clear();
    renderRecordBuffer = nullptr;
    indexBufferAddressBuffer = nullptr;
    pbrBufferAddressBuffer = nullptr;
    geometrySlotCount = 0;
    actualGeometryCount = 0;
    hitGroupVersion++;
    renderVersion++;
    rasterDrawDirtyStartVersion = renderVersion;
    hasPendingGpuUpload = false;
}

void ChunkGeometryMetadataCache::markRasterDrawDirtySlot(uint32_t renderSlot) {
    if (renderSlot == unusedChunkSlot) { return; }
    if (renderSlot >= rasterDrawDirtySlotMarks.size()) {
        rasterDrawDirtySlotMarks.resize(static_cast<size_t>(renderSlot) + 1, 0);
    }
    if (rasterDrawDirtySlotMarks[renderSlot] != 0) { return; }
    if (rasterDrawDirtySlots.empty()) { rasterDrawDirtyStartVersion = renderVersion; }
    rasterDrawDirtySlotMarks[renderSlot] = 1;
    rasterDrawDirtySlots.push_back(renderSlot);
}

void ChunkGeometryMetadataCache::clearRasterDrawDirtySlots() {
    for (uint32_t renderSlot : rasterDrawDirtySlots) {
        if (renderSlot < rasterDrawDirtySlotMarks.size()) { rasterDrawDirtySlotMarks[renderSlot] = 0; }
    }
    rasterDrawDirtySlots.clear();
    rasterDrawDirtyStartVersion = renderVersion;
}

void ChunkGeometryMetadataCache::updateChunk(uint32_t chunkId, const Chunk1 &chunk, uint32_t renderSlot) {
    markRasterDrawDirtySlot(renderSlot);
    if (chunkId >= chunkAllocations.size()) { chunkAllocations.resize(static_cast<size_t>(chunkId) + 1); }

    const uint32_t geometryCount = chunk.geometryCount;
    const uint32_t geometryCapacity = calculateChunkGeometryCapacity(geometryCount);
    auto &allocation = chunkAllocations[chunkId];
    if (allocation.geometryCapacity != geometryCapacity) {
        if (allocation.allocated()) {
            actualGeometryCount -= allocation.geometryCount;
            clearChunkMetadataSlab(*this, allocation);
            freeSlabsByCapacity[allocation.geometryCapacity].push_back({
                .geometryOffset = allocation.geometryOffset,
                .hitGroupOffset = allocation.hitGroupOffset,
            });
        }

        ChunkMetadataSlabInfo slab = allocateChunkMetadataSlab(*this, geometryCapacity);
        allocation = {
            .geometryOffset = slab.geometryOffset,
            .hitGroupOffset = slab.hitGroupOffset,
            .geometryCapacity = geometryCapacity,
            .geometryCount = 0,
        };
    }

    actualGeometryCount -= allocation.geometryCount;
    allocation.geometryCount = geometryCount;
    actualGeometryCount += allocation.geometryCount;

    hitGroupNames[allocation.hitGroupOffset] =
        chunk.hitGroupNameRefs.empty() ? nullptr : chunk.hitGroupNameRefs.front();
    for (uint32_t i = 0; i < allocation.geometryCapacity; ++i) {
        const bool validGeometry = i < geometryCount;
        const size_t geometryIndex = static_cast<size_t>(allocation.geometryOffset) + i;
        indexBufferAddresses[geometryIndex] =
            validGeometry && chunk.indexBufferAddresses != nullptr ? (*chunk.indexBufferAddresses)[i] : 0;
        pbrBufferAddresses[geometryIndex] =
            validGeometry && chunk.pbrBufferAddresses != nullptr ? (*chunk.pbrBufferAddresses)[i] : 0;

        const size_t hitGroupIndex = static_cast<size_t>(allocation.hitGroupOffset) + i + 1;
        hitGroupNames[hitGroupIndex] =
            validGeometry && i + 1 < chunk.hitGroupNameRefs.size() ? chunk.hitGroupNameRefs[i + 1] : nullptr;
    }

    updateRenderRecord(chunkId, chunk, renderSlot);
    mergeDirtyRange(hitGroupDirtyRanges, allocation.hitGroupOffset, allocation.geometryCapacity + 1);
    mergeDirtyRange(geometryDirtyRanges, allocation.geometryOffset, allocation.geometryCapacity);
    hitGroupVersion++;
    hasPendingGpuUpload = true;
}

void ChunkGeometryMetadataCache::removeChunk(uint32_t chunkId) {
    if (chunkId >= chunkAllocations.size()) { return; }

    auto &allocation = chunkAllocations[chunkId];
    if (!allocation.allocated()) { return; }

    actualGeometryCount -= allocation.geometryCount;
    clearChunkMetadataSlab(*this, allocation);
    freeSlabsByCapacity[allocation.geometryCapacity].push_back({
        .geometryOffset = allocation.geometryOffset,
        .hitGroupOffset = allocation.hitGroupOffset,
    });
    allocation = ChunkMetadataSlabAllocationInfo{};
    hitGroupVersion++;
    renderVersion++;
    hasPendingGpuUpload = true;
}

void ChunkGeometryMetadataCache::updateRenderRecord(uint32_t chunkId, const Chunk1 &chunk, uint32_t renderSlot) {
    markRasterDrawDirtySlot(renderSlot);
    if (chunkId >= chunkAllocations.size()) { return; }

    const auto &allocation = chunkAllocations[chunkId];
    if (!allocation.allocated()) { return; }

    if (renderRecords.size() <= renderSlot) { renderRecords.resize(static_cast<size_t>(renderSlot) + 1); }
    renderRecords[renderSlot] = {
        .positionAndGeometryCount =
            glm::ivec4(chunk.x, chunk.y, chunk.z, static_cast<int32_t>(allocation.geometryCount)),
        .offsetsMaskFlags = glm::uvec4(allocation.geometryOffset, allocation.hitGroupOffset, 0x01, 0),
        .blasAddress = packDeviceAddress(chunk.blas->blasDeviceAddress()),
    };
    mergeDirtyRange(renderRecordDirtyRanges, renderSlot, 1);
    renderVersion++;
    hasPendingGpuUpload = true;
}

void ChunkGeometryMetadataCache::resizeRenderRecords(size_t renderRecordCount) {
    if (renderRecords.size() == renderRecordCount) { return; }
    renderRecords.resize(renderRecordCount);
    if (renderRecords.empty()) {
        hitGroupNames.clear();
        indexBufferAddresses.clear();
        pbrBufferAddresses.clear();
        chunkAllocations.clear();
        freeSlabsByCapacity.clear();
        geometrySlotCount = 0;
        actualGeometryCount = 0;
        hitGroupVersion++;
        renderVersion++;
        renderRecordDirtyRanges.clear();
        geometryDirtyRanges.clear();
        hitGroupDirtyRanges.clear();
        rasterDrawDirtySlots.clear();
        rasterDrawDirtySlotMarks.clear();
    } else {
        renderVersion++;
        rasterDrawDirtySlots.clear();
        rasterDrawDirtySlotMarks.assign(renderRecordCount, 1);
        rasterDrawDirtySlots.reserve(renderRecordCount);
        for (uint32_t renderSlot = 0; renderSlot < renderRecordCount; renderSlot++) {
            rasterDrawDirtySlots.push_back(renderSlot);
        }
    }
    rasterDrawDirtyStartVersion = renderVersion;
    hasPendingGpuUpload = true;
}

static LightData packLight(const LightInfo &light, const glm::vec3 &chunkOrigin) {
    LightData gpuLight{};
    glm::vec3 p0 = light.p0 + chunkOrigin;
    glm::vec3 p1 = light.p1 + chunkOrigin;
    glm::vec3 p2 = light.p2 + chunkOrigin;
    glm::vec3 p3 = light.p3 + chunkOrigin;

    gpuLight.p0Area = glm::vec4(p0, light.area);
    gpuLight.p1 = glm::vec4(p1, 0.0f);
    gpuLight.p2 = glm::vec4(p2, 0.0f);
    gpuLight.p3 = glm::vec4(p3, 0.0f);
    gpuLight.normal = glm::vec4(light.normal, 0.0f);
    gpuLight.radiance = glm::vec4(light.radiance, light.restirBoost);
    uint32_t stableIDLow = static_cast<uint32_t>(light.stableID & 0xffffffffull);
    uint32_t stableIDHigh = static_cast<uint32_t>((light.stableID >> 32u) & 0xffffffffull);
    gpuLight.sourceIDData = glm::vec4(packUintBits(stableIDLow), packUintBits(stableIDHigh), 0.0f, 0.0f);
    return gpuLight;
}

void ChunkBuildData::buildLightBuffer(const std::shared_ptr<vk::VMA> &vma,
                                      const std::shared_ptr<vk::Device> &device,
                                      bool persistStaging) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildData.buildLightBuffer");
    lightBuffer = nullptr;
    lightCount = 0;

    if (!collectChunkEmission) {
        (void)vma;
        (void)device;
        return;
    }

    if (lightInfos.empty()) { return; }

    std::vector<LightData> gpuLights;
    gpuLights.reserve(lightInfos.size());

    glm::vec3 chunkOrigin(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    for (const auto &light : lightInfos) {
        float maxRadiance = std::max({light.radiance.r, light.radiance.g, light.radiance.b});
        if (light.area <= 1e-6f || maxRadiance <= 1e-6f) { continue; }

        gpuLights.push_back(packLight(light, chunkOrigin));
    }

    if (gpuLights.empty()) { return; }

    lightCount = static_cast<uint32_t>(gpuLights.size());

    size_t lightBytes = static_cast<size_t>(lightCount) * sizeof(LightData);
    lightBuffer =
        vk::DeviceLocalBuffer::create(vma, device, persistStaging, lightBytes,
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 16);
    lightBuffer->setMemoryCategory(vk::ResourceMemoryCategory::Chunk);
#ifdef DEBUG
    if ((lightBuffer->bufferAddress() & (16 - 1)) != 0) {
        throw std::runtime_error("Chunk light buffer address is not 16-byte aligned");
    }
#endif
    lightBuffer->uploadToStagingBuffer(gpuLights.data(), lightBytes, 0);
}

ChunkBuildData::ChunkBuildData(int64_t id,
                               int x,
                               int y,
                               int z,
                               int64_t version,
                               bool collectChunkEmission,
                               bool important,
                               uint32_t allVertexCount,
                               uint32_t allIndexCount,
                               uint32_t geometryCount,
                               std::vector<World::GeometryTypes> &&geometryTypes,
                               std::vector<std::string> &&geometryGroupNames,
                               std::vector<ChunkBuildGeometry> &&geometries)
    : id(id),
      x(x),
      y(y),
      z(z),
      version(version),
      collectChunkEmission(collectChunkEmission),
      important(important),
      allVertexCount(allVertexCount),
      allIndexCount(allIndexCount),
      geometryCount(geometryCount),
      geometryTypes(std::move(geometryTypes)),
      geometryGroupNames(std::move(geometryGroupNames)),
      geometries(std::move(geometries)),
      indexBufferAddresses(),
      pbrBufferAddresses(),
      indexBuffer(nullptr),
      pbrBuffer(nullptr),
      blasStorageBuffer(nullptr),
      blasScratchBuffer(nullptr),
      blas(nullptr),
      blasBuilder(nullptr),
      indexBufferPool(nullptr),
      pbrBufferPool(nullptr),
      blasStoragePool(nullptr),
      blasScratchPool(nullptr),
      lightInfos(),
      lightBuffer(nullptr),
      lightCount(0) {}

void ChunkBuildData::buildLightInfos(const Emission &emission) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildData.buildLightInfos");
    lightInfos.clear();

    if (!collectChunkEmission) {
        (void)emission;
        return;
    }

    std::unordered_map<uint32_t, Emission::TextureOccupancy> textureOccupancyCache;

    for (uint32_t geometryIndex = 0; geometryIndex < geometries.size(); geometryIndex++) {
        const auto &geometry = geometries[geometryIndex];
        if (geometry.vertexCount < 4) { continue; }
        const auto *geometryVertices = geometry.vertexData();

        uint32_t quadIndex = 0;
        for (uint32_t vertexIndex = 0; vertexIndex + 3 < geometry.vertexCount; vertexIndex += 4, quadIndex++) {
            const auto &v0 = geometryVertices[vertexIndex + 0];
            const auto &v1 = geometryVertices[vertexIndex + 1];
            const auto &v2 = geometryVertices[vertexIndex + 2];
            const auto &v3 = geometryVertices[vertexIndex + 3];
            const glm::vec3 p0 = vk::Vertex::position(v0);
            const glm::vec3 p1 = vk::Vertex::position(v1);
            const glm::vec3 p2 = vk::Vertex::position(v2);
            const glm::vec3 p3 = vk::Vertex::position(v3);

            if (!vk::Vertex::hasFlag(v0.packedData, vk::Vertex::useTextureBit) ||
                !vk::Vertex::hasFlag(v1.packedData, vk::Vertex::useTextureBit) ||
                !vk::Vertex::hasFlag(v2.packedData, vk::Vertex::useTextureBit) ||
                !vk::Vertex::hasFlag(v3.packedData, vk::Vertex::useTextureBit)) {
                continue;
            }
            const uint32_t textureID = v0.textureID;

            std::array<glm::vec2, 4> quadUv = {v0.textureUV, v1.textureUV, v2.textureUV, v3.textureUV};
            glm::vec2 uvMin(std::numeric_limits<float>::max());
            glm::vec2 uvMax(std::numeric_limits<float>::lowest());
            for (glm::vec2 uv : quadUv) {
                uvMin = glm::min(uvMin, uv);
                uvMax = glm::max(uvMax, uv);
            }

            glm::vec2 uvSize = uvMax - uvMin;
            if (uvSize.x <= 1e-6f || uvSize.y <= 1e-6f) { continue; }

            auto [occupancyIter, inserted] = textureOccupancyCache.try_emplace(textureID);
            if (inserted) { emission.copyTextureOccupancy(textureID, occupancyIter->second); }
            if (!Emission::textureOccupancyOverlaps(occupancyIter->second, uvMin, uvMax)) { continue; }

            glm::vec3 tint0 = emissionVertexTint(v0);
            glm::vec3 tint1 = emissionVertexTint(v1);
            glm::vec3 tint2 = emissionVertexTint(v2);
            glm::vec3 tint3 = emissionVertexTint(v3);

            glm::vec3 triNormal0 = glm::cross(p1 - p0, p2 - p0);
            glm::vec3 triNormal1 = glm::cross(p2 - p0, p3 - p0);
            glm::vec3 baseNormal = triNormal0 + triNormal1;
            float baseNormalLength = glm::length(baseNormal);
            if (baseNormalLength <= 1e-6f) { continue; }

            glm::vec3 normal = baseNormal / baseNormalLength;
            float quadArea = triangleArea(p0, p1, p2) + triangleArea(p0, p2, p3);
            if (quadArea <= 1e-6f) { continue; }

            thread_local std::vector<std::shared_ptr<const EmissionCell>> candidateCells;
            emission.collectCells(textureID, uvMin, uvMax, candidateCells);
            for (const auto &candidateCell : candidateCells) {
                if (candidateCell == nullptr) { continue; }

                const auto &cell = *candidateCell;
                glm::vec2 overlapMin = glm::max(uvMin, cell.uvMin);
                glm::vec2 overlapMax = glm::min(uvMax, cell.uvMax);
                glm::vec2 overlapSize = overlapMax - overlapMin;
                if (overlapSize.x <= 1e-6f || overlapSize.y <= 1e-6f || cell.avgEmission <= 0.0f) { continue; }

                uint64_t baseStableID = buildChunkLightStableID(id, geometryIndex, quadIndex, cell.stableKey);
                std::array<std::array<glm::vec2, 3>, 2> triangleUv = {{
                    {v0.textureUV, v1.textureUV, v2.textureUV},
                    {v0.textureUV, v2.textureUV, v3.textureUV},
                }};
                std::array<std::array<glm::vec3, 3>, 2> trianglePos = {{
                    {p0, p1, p2},
                    {p0, p2, p3},
                }};
                std::array<std::array<glm::vec3, 3>, 2> triangleTint = {{
                    {tint0, tint1, tint2},
                    {tint0, tint2, tint3},
                }};

                const size_t logicalLightStart = lightInfos.size();
                float logicalLightArea = 0.0f;
                uint64_t trianglePieceIndex = 0;
                for (int triangleIndex = 0; triangleIndex < 2; ++triangleIndex) {
                    std::vector<glm::vec2> clippedPolygon =
                        clipUvTriangleToRect(triangleUv[triangleIndex], overlapMin, overlapMax);
                    if (clippedPolygon.size() < 3) { continue; }

                    glm::vec3 fanOrigin;
                    glm::vec3 fanOriginTint;
                    if (!uvToTrianglePosition(clippedPolygon[0], triangleUv[triangleIndex], trianglePos[triangleIndex],
                                              fanOrigin) ||
                        !uvToTriangleTint(clippedPolygon[0], triangleUv[triangleIndex], triangleTint[triangleIndex],
                                          fanOriginTint)) {
                        continue;
                    }

                    for (size_t polygonIndex = 1; polygonIndex + 1 < clippedPolygon.size(); ++polygonIndex) {
                        glm::vec3 p1;
                        glm::vec3 p2;
                        glm::vec3 p1Tint;
                        glm::vec3 p2Tint;
                        if (!uvToTrianglePosition(clippedPolygon[polygonIndex], triangleUv[triangleIndex],
                                                  trianglePos[triangleIndex], p1) ||
                            !uvToTriangleTint(clippedPolygon[polygonIndex], triangleUv[triangleIndex],
                                              triangleTint[triangleIndex], p1Tint) ||
                            !uvToTrianglePosition(clippedPolygon[polygonIndex + 1], triangleUv[triangleIndex],
                                                  trianglePos[triangleIndex], p2) ||
                            !uvToTriangleTint(clippedPolygon[polygonIndex + 1], triangleUv[triangleIndex],
                                              triangleTint[triangleIndex], p2Tint)) {
                            continue;
                        }

                        float triangleLightArea = triangleArea(fanOrigin, p1, p2);
                        if (triangleLightArea <= 1e-6f) { continue; }

                        glm::vec3 lightNormal = glm::cross(p1 - fanOrigin, p2 - fanOrigin);
                        float lightNormalLength = glm::length(lightNormal);
                        lightNormal = lightNormalLength > 1e-6f ? lightNormal / lightNormalLength : normal;
                        glm::vec3 avgTint = glm::max((fanOriginTint + p1Tint + p2Tint) / 3.0f, glm::vec3(0.0f));

                        LightInfo info{
                            .p0 = fanOrigin,
                            .p1 = p1,
                            .p2 = p2,
                            .p3 = p2,
                            .normal = lightNormal,
                            .radiance = cell.avgColor * avgTint * cell.avgEmission,
                            .area = triangleLightArea,
                            .textureID = v0.textureID,
                            .stableID = hashCombine64(baseStableID, trianglePieceIndex++),
                        };
                        lightInfos.push_back(info);
                        logicalLightArea += triangleLightArea;
                    }
                }
                const float restirBoost = calculateRestirLightBoost(logicalLightArea);
                for (size_t lightIndex = logicalLightStart; lightIndex < lightInfos.size(); lightIndex++) {
                    lightInfos[lightIndex].restirBoost = restirBoost;
                }
            }
        }
    }
}

void ChunkBuildData::build(bool persistStaging) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildData.build");
    MCVR_PROFILE_METRIC("chunk_build.geometry_count", geometryCount);
    MCVR_PROFILE_METRIC("chunk_build.vertex_count", allVertexCount);
    MCVR_PROFILE_METRIC("chunk_build.index_count", allIndexCount);
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto chunks = Renderer::instance().world() != nullptr ? Renderer::instance().world()->chunks() : nullptr;
    const uint64_t currentFrameSerial = chunks != nullptr ? chunks->frameSerial() : 0;

    buildLightBuffer(vma, device, persistStaging);

    if (geometryCount == 0) {
        blas = nullptr;
        blasBuilder = nullptr;
        releaseBuildInputs();
        return;
    }

    std::vector<uint32_t> geometryVertexOffsets;
    std::vector<uint32_t> geometryIndexOffsets;
    geometryVertexOffsets.reserve(geometryCount);
    geometryIndexOffsets.reserve(geometryCount);

    uint32_t totalVertexCount = 0;
    uint32_t totalIndexCount = 0;
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.prepareGeometryOffsets");
        for (int i = 0; i < geometryCount; i++) {
            geometryVertexOffsets.push_back(totalVertexCount);
            geometryIndexOffsets.push_back(totalIndexCount);
            totalVertexCount += geometries[i].vertexCount;
            totalIndexCount += geometries[i].indexCount;
        }
    }

    const VkBufferUsageFlags pbrUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    const VkBufferUsageFlags indexUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    const VkDeviceSize pbrBytes =
        static_cast<VkDeviceSize>(totalVertexCount) * sizeof(vk::VertexFormat::ChunkPBRVertex);
    const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(totalIndexCount) * sizeof(uint32_t);
    MCVR_PROFILE_METRIC("chunk_build.pbr_bytes", pbrBytes);
    MCVR_PROFILE_METRIC("chunk_build.index_bytes", indexBytes);
    if (pbrBufferPool == nullptr || indexBufferPool == nullptr) {
        throw std::runtime_error("chunk geometry buffer pool is not initialized");
    }
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.acquireGeometryBuffers");
        pbrBuffer = pbrBufferPool->acquire(vma, device, currentFrameSerial, pbrBytes, pbrUsage, persistStaging,
                                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        indexBuffer = indexBufferPool->acquire(vma, device, currentFrameSerial, indexBytes, indexUsage, persistStaging,
                                               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }

    std::vector<vk::VertexFormat::ChunkPBRVertex> packedVertices;
    std::vector<uint32_t> packedIndices;
    packedVertices.reserve(totalVertexCount);
    packedIndices.reserve(totalIndexCount);
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.packGeometry");
        buildPackedGeometry(geometries, packedVertices, packedIndices);
    }

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.uploadGeometryStaging");
        if (!packedVertices.empty()) {
            pbrBuffer->uploadToStagingBuffer(packedVertices.data(),
                                             packedVertices.size() * sizeof(vk::VertexFormat::ChunkPBRVertex), 0);
        }
        if (!packedIndices.empty()) {
            indexBuffer->uploadToStagingBuffer(packedIndices.data(), packedIndices.size() * sizeof(uint32_t), 0);
        }
    }

    indexBufferAddresses.reserve(geometryCount);
    pbrBufferAddresses.reserve(geometryCount);

    blasBuilder = vk::BLASBuilder::create();
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.defineBlasGeometry");
        auto blasGeometryBuilder = blasBuilder->beginGeometries();
        for (int i = 0; i < geometryCount; i++) {
            const VkDeviceAddress geometryIndexAddress =
                indexBuffer->bufferAddress() + geometryIndexOffsets[i] * sizeof(uint32_t);
            const VkDeviceAddress geometryPbrAddress =
                pbrBuffer->bufferAddress() + geometryVertexOffsets[i] * sizeof(vk::VertexFormat::ChunkPBRVertex);

            indexBufferAddresses.push_back(geometryIndexAddress);
            pbrBufferAddresses.push_back(geometryPbrAddress);

            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::ChunkPBRVertex>(
                geometryPbrAddress, geometries[i].vertexCount, geometryIndexAddress, geometries[i].indexCount,
                geometryTypes[i] == World::WORLD_SOLID);
        }
        blasGeometryBuilder->endGeometries();
    }

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.flushGeometryStaging");
        pbrBuffer->flushStagingBuffer();
        indexBuffer->flushStagingBuffer();
    }

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.queryBlasSize");
        blasBuilder = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                          ->querySizeInfo(device);
    }
    if (blasStoragePool == nullptr || blasScratchPool == nullptr) {
        throw std::runtime_error("chunk BLAS buffer pool is not initialized");
    }
    MCVR_PROFILE_METRIC("chunk_build.blas_storage_bytes", blasBuilder->requiredStorageSize());
    MCVR_PROFILE_METRIC("chunk_build.blas_scratch_bytes", blasBuilder->requiredScratchSize());
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.acquireBlasBuffers");
        blasStorageBuffer = blasStoragePool->acquire(
            vma, device, currentFrameSerial, blasBuilder->requiredStorageSize(),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false,
            VMA_MEMORY_USAGE_GPU_ONLY, 256);
        blasScratchBuffer = blasScratchPool->acquire(
            vma, device, currentFrameSerial, blasBuilder->requiredScratchSize(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false,
            VMA_MEMORY_USAGE_GPU_ONLY,
            physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment);
    }
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildData.buildBlas");
        blas = blasBuilder->useExternalStorage(blasStorageBuffer, 0, blasBuilder->requiredStorageSize())
                   ->useExternalScratch(blasScratchBuffer)
                   ->build(device);
    }

    releaseBuildInputs();
}

void ChunkBuildData::releaseBuildInputs() {
    releaseBuildGeometryInputs(geometries);
}

ChunkBuildDataBatch::ChunkBuildDataBatch(std::vector<std::shared_ptr<ChunkBuildData>> &&batchData)
    : batchData(std::move(batchData)) {}

ChunkBuildDataBatch::ChunkBuildDataBatch(uint32_t maxBatchSize,
                                         std::set<int64_t> &queuedIndexSet,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         glm::vec3 cameraPos) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.selectQueued");
    MCVR_PROFILE_METRIC("chunk_batch.queued_count", queuedIndexSet.size());
    MCVR_PROFILE_METRIC("chunk_batch.max_batch_size", maxBatchSize);
    if (maxBatchSize == 0 || queuedIndexSet.empty()) { return; }

    auto currentTime = std::chrono::steady_clock::now();
    struct Candidate {
        int64_t id = -1;
        float score = 0.0f;
    };
    std::vector<Candidate> importantCandidates;
    std::vector<Candidate> normalCandidates;
    importantCandidates.reserve(maxBatchSize);
    normalCandidates.reserve(maxBatchSize);

    auto insertCandidate = [&](std::vector<Candidate> &candidates, Candidate candidate) {
        if (candidates.size() < maxBatchSize) {
            candidates.push_back(candidate);
        } else if (candidate.score > candidates.back().score) {
            candidates.back() = candidate;
        } else {
            return;
        }
        for (size_t i = candidates.size() - 1; i > 0; i--) {
            if (candidates[i - 1].score >= candidates[i].score) { break; }
            std::swap(candidates[i - 1], candidates[i]);
        }
    };

    size_t consideredCount = 0;
    size_t consideredImportantCount = 0;
    size_t consideredNormalCount = 0;
    size_t prunedCount = 0;
    for (auto iter = queuedIndexSet.begin(); iter != queuedIndexSet.end();) {
        const int64_t queuedIndex = *iter;
        if (queuedIndex < 0 || static_cast<size_t>(queuedIndex) >= chunks.size() ||
            static_cast<size_t>(queuedIndex) >= chunkBuildDatas.size()) {
            iter = queuedIndexSet.erase(iter);
            prunedCount++;
            continue;
        }

        const auto &chunk = chunks[queuedIndex];
        const auto &chunkBuildData = chunkBuildDatas[queuedIndex];
        if (chunk == nullptr || chunkBuildData == nullptr || chunkBuildData->version + 1 != chunk->latestVersion) {
            iter = queuedIndexSet.erase(iter);
            prunedCount++;
            continue;
        }

        glm::vec3 chunkPos = {
            static_cast<float>(chunkBuildData->x),
            static_cast<float>(chunkBuildData->y),
            static_cast<float>(chunkBuildData->z),
        };

        auto &candidates = chunkBuildData->important ? importantCandidates : normalCandidates;
        insertCandidate(candidates, {
                                        .id = queuedIndex,
                                        .score = chunk->buildFactor(currentTime, cameraPos, chunkPos),
                                    });
        consideredCount++;
        if (chunkBuildData->important) {
            consideredImportantCount++;
        } else {
            consideredNormalCount++;
        }
        ++iter;
    }
    MCVR_PROFILE_METRIC("chunk_batch.considered_count", consideredCount);
    MCVR_PROFILE_METRIC("chunk_batch.considered_important_count", consideredImportantCount);
    MCVR_PROFILE_METRIC("chunk_batch.considered_normal_count", consideredNormalCount);
    MCVR_PROFILE_METRIC("chunk_batch.pruned_stale_count", prunedCount);
    MCVR_PROFILE_METRIC("chunk_batch.important_candidates", importantCandidates.size());
    MCVR_PROFILE_METRIC("chunk_batch.normal_candidates", normalCandidates.size());

    auto selectCandidate = [&](const Candidate &candidate) {
        const int64_t queuedIndex = candidate.id;
        auto iter = queuedIndexSet.find(queuedIndex);
        if (iter != queuedIndexSet.end()) { queuedIndexSet.erase(iter); }

        auto data = chunkBuildDatas[queuedIndex];
        chunkBuildDatas[queuedIndex] = nullptr;
        batchData.push_back(data);
    };

    static constexpr int prioritizedTaskBurst = 2;
    int remainingPrioritizableTasks = prioritizedTaskBurst;
    size_t importantIndex = 0;
    size_t normalIndex = 0;
    size_t selectedImportantCount = 0;
    size_t selectedNormalCount = 0;

    while (batchData.size() < maxBatchSize &&
           (importantIndex < importantCandidates.size() || normalIndex < normalCandidates.size())) {
        const bool hasImportant = importantIndex < importantCandidates.size();
        const bool hasNormal = normalIndex < normalCandidates.size();
        bool takeImportant = false;

        if (hasImportant) {
            takeImportant =
                !hasNormal || (remainingPrioritizableTasks > 0 &&
                               importantCandidates[importantIndex].score >= normalCandidates[normalIndex].score);
        }

        if (takeImportant) {
            selectCandidate(importantCandidates[importantIndex++]);
            remainingPrioritizableTasks--;
            selectedImportantCount++;
        } else if (hasNormal) {
            selectCandidate(normalCandidates[normalIndex++]);
            remainingPrioritizableTasks = prioritizedTaskBurst;
            selectedNormalCount++;
        }
    }

    MCVR_PROFILE_METRIC("chunk_batch.selected_candidates", batchData.size());
    MCVR_PROFILE_METRIC("chunk_batch.selected_important_count", selectedImportantCount);
    MCVR_PROFILE_METRIC("chunk_batch.selected_normal_count", selectedNormalCount);
}

void ChunkBuildDataBatch::build() {
    MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.build");
    MCVR_PROFILE_METRIC("chunk_batch.size", batchData.size());
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto chunks = Renderer::instance().world() != nullptr ? Renderer::instance().world()->chunks() : nullptr;
    const uint64_t currentFrameSerial = chunks != nullptr ? chunks->frameSerial() : 0;

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.buildLightBuffers");
        for (auto &data : batchData) {
            if (data == nullptr) { continue; }
            data->buildLightBuffer(vma, device, true);
        }
    }

    std::vector<uint32_t> instanceOffsets;
    std::vector<uint32_t> geometryVertexOffsets;
    std::vector<uint32_t> geometryIndexOffsets;
    instanceOffsets.reserve(batchData.size());

    uint32_t totalGeometryCount = 0;
    uint32_t totalVertexCount = 0;
    uint32_t totalIndexCount = 0;

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.countGeometry");
        for (const auto &data : batchData) {
            if (data == nullptr) {
                instanceOffsets.push_back(totalGeometryCount);
                continue;
            }
            instanceOffsets.push_back(totalGeometryCount);
            for (int i = 0; i < data->geometryCount; i++) {
                geometryVertexOffsets.push_back(totalVertexCount);
                geometryIndexOffsets.push_back(totalIndexCount);
                totalVertexCount += data->geometries[i].vertexCount;
                totalIndexCount += data->geometries[i].indexCount;
            }
            totalGeometryCount += data->geometryCount;
        }
    }
    MCVR_PROFILE_METRIC("chunk_batch.geometry_count", totalGeometryCount);
    MCVR_PROFILE_METRIC("chunk_batch.vertex_count", totalVertexCount);
    MCVR_PROFILE_METRIC("chunk_batch.index_count", totalIndexCount);

    if (totalGeometryCount == 0) {
        for (const auto &data : batchData) {
            data->indexBufferAddresses.clear();
            data->pbrBufferAddresses.clear();
            data->indexBuffer = nullptr;
            data->pbrBuffer = nullptr;
            data->blas = nullptr;
            data->blasBuilder = nullptr;
            releaseChunkBuildDataInputs(data);
        }
        return;
    }

    const VkBufferUsageFlags pbrUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    const VkBufferUsageFlags indexUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    pbrBuffer = nullptr;
    indexBuffer = nullptr;

    blasBuilders.clear();
    blasBuilders.reserve(batchData.size());
    std::vector<uint32_t> nonEmptyInstanceIndices;
    nonEmptyInstanceIndices.reserve(batchData.size());

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.buildInstances");
        for (int chunkIndex = 0; auto &data : batchData) {
            if (data == nullptr) {
                chunkIndex++;
                continue;
            }
            data->indexBufferAddresses.clear();
            data->pbrBufferAddresses.clear();
            data->indexBufferAddresses.reserve(data->geometryCount);
            data->pbrBufferAddresses.reserve(data->geometryCount);

            if (data->geometryCount == 0) {
                data->indexBuffer = nullptr;
                data->pbrBuffer = nullptr;
                data->blas = nullptr;
                data->blasBuilder = nullptr;
                chunkIndex++;
                continue;
            }

            const VkDeviceSize pbrBytes =
                static_cast<VkDeviceSize>(data->allVertexCount) * sizeof(vk::VertexFormat::ChunkPBRVertex);
            const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(data->allIndexCount) * sizeof(uint32_t);
            if (data->pbrBufferPool == nullptr || data->indexBufferPool == nullptr) {
                throw std::runtime_error("chunk geometry buffer pool is not initialized");
            }
            {
                MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.acquireGeometryBuffers");
                data->pbrBuffer = data->pbrBufferPool->acquire(vma, device, currentFrameSerial, pbrBytes, pbrUsage,
                                                               true, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
                data->indexBuffer = data->indexBufferPool->acquire(
                    vma, device, currentFrameSerial, indexBytes, indexUsage, true, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            }

            std::vector<vk::VertexFormat::ChunkPBRVertex> packedVertices;
            std::vector<uint32_t> packedIndices;
            packedVertices.reserve(data->allVertexCount);
            packedIndices.reserve(data->allIndexCount);
            {
                MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.packGeometry");
                buildPackedGeometry(data->geometries, packedVertices, packedIndices);
            }
            {
                MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.uploadGeometryStaging");
                if (!packedVertices.empty()) {
                    data->pbrBuffer->uploadToStagingBuffer(
                        packedVertices.data(), packedVertices.size() * sizeof(vk::VertexFormat::ChunkPBRVertex), 0);
                }
                if (!packedIndices.empty()) {
                    data->indexBuffer->uploadToStagingBuffer(packedIndices.data(),
                                                             packedIndices.size() * sizeof(uint32_t), 0);
                }
                data->pbrBuffer->flushStagingBuffer();
                data->indexBuffer->flushStagingBuffer();
            }

            auto blasBuilder = vk::BLASBuilder::create();
            {
                MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.defineBlasGeometry");
                auto blasGeometryBuilder = blasBuilder->beginGeometries();

                uint32_t geometryVertexOffset = 0;
                uint32_t geometryIndexOffset = 0;
                for (int i = 0; i < data->geometryCount; i++) {
                    const VkDeviceAddress geometryIndexAddress =
                        data->indexBuffer->bufferAddress() + geometryIndexOffset * sizeof(uint32_t);
                    const VkDeviceAddress geometryPbrAddress =
                        data->pbrBuffer->bufferAddress() +
                        geometryVertexOffset * sizeof(vk::VertexFormat::ChunkPBRVertex);

                    data->indexBufferAddresses.push_back(geometryIndexAddress);
                    data->pbrBufferAddresses.push_back(geometryPbrAddress);

                    blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::ChunkPBRVertex>(
                        geometryPbrAddress, data->geometries[i].vertexCount, geometryIndexAddress,
                        data->geometries[i].indexCount, data->geometryTypes[i] == World::WORLD_SOLID);
                    geometryVertexOffset += data->geometries[i].vertexCount;
                    geometryIndexOffset += data->geometries[i].indexCount;
                }

                blasGeometryBuilder->endGeometries();
            }
            {
                MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.queryBlasSize");
                data->blasBuilder =
                    blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                        ->querySizeInfo(device);
            }
            blasBuilders.push_back(data->blasBuilder);
            nonEmptyInstanceIndices.push_back(chunkIndex);
            chunkIndex++;
        }
    }

    const VkDeviceSize scratchAlignment =
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment;
    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.buildBlasInstances");
        MCVR_PROFILE_METRIC("chunk_batch.non_empty_instances", nonEmptyInstanceIndices.size());
        for (int i = 0; i < nonEmptyInstanceIndices.size(); i++) {
            auto &data = batchData[nonEmptyInstanceIndices[i]];
            if (data->blasStoragePool == nullptr || data->blasScratchPool == nullptr) {
                throw std::runtime_error("chunk BLAS buffer pool is not initialized");
            }
            {
                MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.acquireBlasBuffers");
                data->blasStorageBuffer = data->blasStoragePool->acquire(
                    vma, device, currentFrameSerial, data->blasBuilder->requiredStorageSize(),
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    false, VMA_MEMORY_USAGE_GPU_ONLY, 256);
                data->blasScratchBuffer = data->blasScratchPool->acquire(
                    vma, device, currentFrameSerial, data->blasBuilder->requiredScratchSize(),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false,
                    VMA_MEMORY_USAGE_GPU_ONLY, scratchAlignment);
            }
            data->blasBuilder->useExternalStorage(data->blasStorageBuffer, 0, data->blasBuilder->requiredStorageSize())
                ->useExternalScratch(data->blasScratchBuffer);
            data->blas = data->blasBuilder->build(device);
        }
    }

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildDataBatch.releaseBuildInputs");
        for (const auto &data : batchData) { releaseChunkBuildDataInputs(data); }
    }
}

ChunkBuildScheduler::ChunkBuildScheduler(std::set<int64_t> &queuedIndex,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         std::recursive_mutex &mutex,
                                         std::vector<ChunkPackedData> &chunkPackedData,
                                         std::vector<std::shared_ptr<Chunk1>> &renderableChunks,
                                         std::vector<std::shared_ptr<ChunkRenderData>> &renderableChunkSnapshots,
                                         std::vector<int64_t> &renderableChunkIds,
                                         std::vector<uint32_t> &renderableChunkSlots,
                                         ChunkGeometryMetadataCache &chunkGeometryMetadataCache,
                                         uint32_t chunkBuildingBatchSize,
                                         uint32_t chunkBuildingTotalBatches,
                                         uint32_t chunkBuildingThreads)
    : queuedIndex_(queuedIndex),
      chunks_(chunks),
      chunkBuildDatas_(chunkBuildDatas),
      mutex_(mutex),
      chunkPackedData_(chunkPackedData),
      renderableChunks_(renderableChunks),
      renderableChunkSnapshots_(renderableChunkSnapshots),
      renderableChunkIds_(renderableChunkIds),
      renderableChunkSlots_(renderableChunkSlots),
      chunkGeometryMetadataCache_(chunkGeometryMetadataCache),
      chunkBuildingBatchSize_(chunkBuildingBatchSize),
      chunkBuildingTotalBatches_(chunkBuildingTotalBatches),
      chunkBuildingThreads_(chunkBuildingThreads) {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    useSecondaryQueue_ = device->hasIndependentSecondaryQueue();
    auto commandPool = useSecondaryQueue_ ? framework->asyncCommandPool() : framework->mainCommandPool();

    uint32_t numFences = chunkBuildingTotalBatches_;
    for (int i = 0; i < numFences; i++) {
        freeFences_.push(vk::Fence::create(device));
        freeCommandBuffers_.push(vk::CommandBuffer::create(device, commandPool));
    }

    startCpuBuildWorkers();
}

ChunkBuildScheduler::~ChunkBuildScheduler() {
    stopCpuBuildWorkers();
}

void ChunkBuildScheduler::startCpuBuildWorkers() {
    if (chunkBuildingTotalBatches_ == 0) { return; }

    const uint32_t workerCount = std::min(chunkBuildingThreads_, chunkBuildingTotalBatches_);
    cpuBuildWorkers_.reserve(workerCount);
    for (uint32_t workerIndex = 0; workerIndex < workerCount; workerIndex++) {
        cpuBuildWorkers_.emplace_back([this]() { cpuBuildWorkerMain(); });
    }
}

void ChunkBuildScheduler::stopCpuBuildWorkers() {
    {
        std::lock_guard<std::mutex> lock(cpuBuildMutex_);
        stopCpuBuildWorkers_ = true;
        pendingCpuBuildJobs_.clear();
    }
    cpuBuildCondition_.notify_all();

    for (auto &worker : cpuBuildWorkers_) {
        if (worker.joinable()) { worker.join(); }
    }
    cpuBuildWorkers_.clear();

    {
        std::lock_guard<std::mutex> lock(cpuBuildMutex_);
        readyCpuBuildJobs_.clear();
        activeCpuBuildJobCount_ = 0;
    }
}

void ChunkBuildScheduler::cpuBuildWorkerMain() {
    setCurrentChunkBuildThreadLowPriority();

    for (;;) {
        CpuBuildJob job;
        {
            std::unique_lock<std::mutex> lock(cpuBuildMutex_);
            cpuBuildCondition_.wait(lock, [this]() { return stopCpuBuildWorkers_ || !pendingCpuBuildJobs_.empty(); });
            if (stopCpuBuildWorkers_) { return; }

            job = std::move(pendingCpuBuildJobs_.front());
            pendingCpuBuildJobs_.pop_front();
            activeCpuBuildJobCount_++;
        }

        try {
            MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.cpuBuildWorker");
            job.chunkBuildDataBatch->build();
        } catch (...) { job.exception = std::current_exception(); }

        {
            std::lock_guard<std::mutex> lock(cpuBuildMutex_);
            activeCpuBuildJobCount_--;
            readyCpuBuildJobs_.push_back(std::move(job));
        }
        cpuBuildCondition_.notify_all();
    }
}

void ChunkBuildScheduler::enqueueCpuBuildJob(CpuBuildJob job) {
    {
        std::lock_guard<std::mutex> lock(cpuBuildMutex_);
        if (stopCpuBuildWorkers_) { return; }
        pendingCpuBuildJobs_.push_back(std::move(job));
    }
    cpuBuildCondition_.notify_one();
}

bool ChunkBuildScheduler::hasCpuBuildWorkLocked() const {
    return !pendingCpuBuildJobs_.empty() || !readyCpuBuildJobs_.empty() || activeCpuBuildJobCount_ > 0;
}

bool ChunkBuildScheduler::hasCurrentBuildDataLocked(
    const std::shared_ptr<ChunkBuildDataBatch> &chunkBuildDataBatch) const {
    if (chunkBuildDataBatch == nullptr) { return false; }
    for (const auto &chunkBuildData : chunkBuildDataBatch->batchData) {
        if (isBuildDataCurrentLocked(chunkBuildData)) { return true; }
    }
    return false;
}

void ChunkBuildScheduler::submitReadyCpuBuildBatches(uint32_t &batchBudget, bool wait) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.submitReadyCpuBuildBatches");
    MCVR_PROFILE_METRIC("chunk_scheduler.submit_budget", batchBudget);
    MCVR_PROFILE_METRIC("chunk_scheduler.wait_ready", wait ? 1 : 0);
    while (batchBudget > 0) {
        CpuBuildJob job;
        {
            std::unique_lock<std::mutex> lock(cpuBuildMutex_);
            if (wait) {
                cpuBuildCondition_.wait(lock,
                                        [this]() { return !readyCpuBuildJobs_.empty() || !hasCpuBuildWorkLocked(); });
            }
            if (readyCpuBuildJobs_.empty()) { return; }

            job = std::move(readyCpuBuildJobs_.front());
            readyCpuBuildJobs_.pop_front();
        }

        batchBudget--;
        if (job.exception != nullptr) {
            std::unique_lock<std::recursive_mutex> lock(mutex_);
            freeFences_.push(std::move(job.fence));
            freeCommandBuffers_.push(std::move(job.commandBuffer));
            if (job.chunkBuildDataBatch != nullptr) {
                for (const auto &chunkBuildData : job.chunkBuildDataBatch->batchData) {
                    releaseChunkBuildDataBuffersToPools(chunkBuildData, true);
                }
            }
            std::rethrow_exception(job.exception);
        }

        bool hasCurrentBuildData = false;
        {
            std::unique_lock<std::recursive_mutex> lock(mutex_);
            hasCurrentBuildData = hasCurrentBuildDataLocked(job.chunkBuildDataBatch);
            if (!hasCurrentBuildData) {
                freeFences_.push(std::move(job.fence));
                freeCommandBuffers_.push(std::move(job.commandBuffer));
                if (job.chunkBuildDataBatch != nullptr) {
                    for (const auto &chunkBuildData : job.chunkBuildDataBatch->batchData) {
                        releaseChunkBuildDataBuffersToPools(chunkBuildData, true);
                    }
                }
            }
        }
        if (!hasCurrentBuildData) { continue; }

        submitBatch(std::move(job.chunkBuildDataBatch), std::move(job.fence), std::move(job.commandBuffer));
    }
}

void ChunkBuildScheduler::publishBatchLocked(const std::shared_ptr<ChunkBuildDataBatch> &chunkBuildDataBatch) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.publishBatchLocked");
    MCVR_PROFILE_METRIC("chunk_scheduler.publish_batch_size",
                        chunkBuildDataBatch != nullptr ? chunkBuildDataBatch->batchData.size() : 0);
    for (const auto &chunkBuildData : chunkBuildDataBatch->batchData) {
        if (chunkBuildData == nullptr) { continue; }
        if (!isBuildDataCurrentLocked(chunkBuildData)) {
            releaseChunkBuildDataBuffersToPools(chunkBuildData, true);
            continue;
        }

        bool wasEnqueued = chunks_[chunkBuildData->id]->enqueue(chunkBuildData);
        if (wasEnqueued) { releaseChunkBuildDataGeometryStaging(chunkBuildData); }
        releaseChunkBuildDataNonPooledStaging(chunkBuildData);
        releaseChunkBuildDataBuffersToPools(chunkBuildData, !wasEnqueued);
        if (!wasEnqueued) { continue; }

        auto slotUpdateInfo =
            updateRenderableChunkSlot(renderableChunks_, renderableChunkSnapshots_, renderableChunkIds_,
                                      renderableChunkSlots_, chunkBuildData->id, chunks_[chunkBuildData->id]);
        applyRenderableChunkSlotUpdate(chunkGeometryMetadataCache_, chunks_, slotUpdateInfo, chunkBuildData->id,
                                       renderableChunks_.size());
        storeChunkPackedData(chunkPackedData_, chunkBuildData->id, chunkBuildData->x, chunkBuildData->y,
                             chunkBuildData->z, chunkBuildData->geometryCount, chunkBuildData->lightCount,
                             chunkBuildData->lightBuffer != nullptr ? chunkBuildData->lightBuffer->bufferAddress() : 0);
    }
}

bool ChunkBuildScheduler::isBuildDataCurrentLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData) const {
    if (chunkBuildData == nullptr) { return false; }

    const int64_t chunkId = chunkBuildData->id;
    if (chunkId < 0 || static_cast<size_t>(chunkId) >= chunks_.size()) { return false; }

    const auto &chunk = chunks_[chunkId];
    if (chunk == nullptr) { return false; }

    return chunkBuildData->version + 1 == chunk->latestVersion &&
           chunkBuildData->generation == chunk->generation && chunkBuildData->x == chunk->x &&
           chunkBuildData->y == chunk->y && chunkBuildData->z == chunk->z;
}

void ChunkBuildScheduler::pruneStalePendingBatchDataLocked() {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.pruneStalePendingBatchDataLocked");
    MCVR_PROFILE_METRIC("chunk_scheduler.pending_batch_data_before", pendingBatchData_.size());
    auto writeIter = pendingBatchData_.begin();
    for (auto readIter = pendingBatchData_.begin(); readIter != pendingBatchData_.end(); ++readIter) {
        if (!isBuildDataCurrentLocked(*readIter)) { continue; }

        if (writeIter != readIter) { *writeIter = std::move(*readIter); }
        ++writeIter;
    }
    pendingBatchData_.erase(writeIter, pendingBatchData_.end());

    if (pendingBatchData_.empty()) { pendingBatchFrames_ = 0; }
    MCVR_PROFILE_METRIC("chunk_scheduler.pending_batch_data_after", pendingBatchData_.size());
}

void ChunkBuildScheduler::submitBatch(std::shared_ptr<ChunkBuildDataBatch> chunkBuildDataBatch,
                                      std::shared_ptr<vk::Fence> fence,
                                      std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.submitBatch");
    MCVR_PROFILE_METRIC("chunk_scheduler.submit_batch_size",
                        chunkBuildDataBatch != nullptr ? chunkBuildDataBatch->batchData.size() : 0);
    if (!requiresGpuSubmit(chunkBuildDataBatch)) {
        std::unique_lock<std::recursive_mutex> lock(mutex_);
        freeFences_.push(fence);
        freeCommandBuffers_.push(commandBuffer);
        publishBatchLocked(chunkBuildDataBatch);
        return;
    }

    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    const auto queueFamilyIndex =
        useSecondaryQueue_ ? physicalDevice->secondaryQueueIndex() : physicalDevice->mainQueueIndex();
    const VkPipelineStageFlags2 geometryIndexDstStage =
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
        (useSecondaryQueue_ ? 0 :
                              (VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT));
    const VkPipelineStageFlags2 geometryPbrDstStage =
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
        (useSecondaryQueue_ ? 0 :
                              (VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT));
    const VkAccessFlags2 geometryIndexDstAccess =
        VK_ACCESS_2_SHADER_READ_BIT | (useSecondaryQueue_ ? 0 : VK_ACCESS_2_INDEX_READ_BIT);
    const VkAccessFlags2 geometryPbrDstAccess =
        VK_ACCESS_2_SHADER_READ_BIT | (useSecondaryQueue_ ? 0 : VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
    const VkPipelineStageFlags2 lightDstStage =
        useSecondaryQueue_ ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT :
                             (VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    const VkPipelineStageFlags2 blasPostBuildDstStage =
        useSecondaryQueue_ ?
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR :
            (VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.recordBatchCommandBuffer");
        commandBuffer->begin();
        {
            MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.recordGeometryUploads");
            for (const auto &chunkBuildData : chunkBuildDataBatch->batchData) {
                if (chunkBuildData != nullptr && chunkBuildData->pbrBuffer != nullptr) {
                    chunkBuildData->indexBuffer->uploadToBuffer(commandBuffer);
                    chunkBuildData->pbrBuffer->uploadToBuffer(commandBuffer);
                }
                if (chunkBuildData != nullptr && chunkBuildData->lightBuffer != nullptr) {
                    chunkBuildData->lightBuffer->uploadToBuffer(commandBuffer);
                }
            }
        }

        std::vector<vk::CommandBuffer::BufferMemoryBarrier> geometryBarriers;
        {
            MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.recordGeometryBarriers");
            for (const auto &chunkBuildData : chunkBuildDataBatch->batchData) {
                if (chunkBuildData == nullptr || chunkBuildData->pbrBuffer == nullptr) { continue; }
                geometryBarriers.push_back({
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = geometryIndexDstStage,
                    .dstAccessMask = geometryIndexDstAccess,
                    .srcQueueFamilyIndex = queueFamilyIndex,
                    .dstQueueFamilyIndex = queueFamilyIndex,
                    .buffer = chunkBuildData->indexBuffer,
                });
                geometryBarriers.push_back({
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = geometryPbrDstStage,
                    .dstAccessMask = geometryPbrDstAccess,
                    .srcQueueFamilyIndex = queueFamilyIndex,
                    .dstQueueFamilyIndex = queueFamilyIndex,
                    .buffer = chunkBuildData->pbrBuffer,
                });
            }
            if (!geometryBarriers.empty()) { commandBuffer->barriersBufferImage(geometryBarriers, {}); }
        }

        std::vector<vk::CommandBuffer::BufferMemoryBarrier> lightBarriers;
        {
            MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.recordLightBarriers");
            for (const auto &chunkBuildData : chunkBuildDataBatch->batchData) {
                if (chunkBuildData == nullptr || chunkBuildData->lightBuffer == nullptr) { continue; }

                lightBarriers.push_back({
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = lightDstStage,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .srcQueueFamilyIndex = queueFamilyIndex,
                    .dstQueueFamilyIndex = queueFamilyIndex,
                    .buffer = chunkBuildData->lightBuffer,
                });
            }
            if (!lightBarriers.empty()) { commandBuffer->barriersBufferImage(lightBarriers, {}); }
        }

        if (!chunkBuildDataBatch->blasBuilders.empty()) {
            MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.recordBatchBlasBuild");
            vk::BLASBuilder::batchSubmit(chunkBuildDataBatch->blasBuilders, commandBuffer, chunkBuildDataBatch);
            commandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
                .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .srcAccessMask =
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                .dstStageMask = blasPostBuildDstStage,
                .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            }});
        }
        commandBuffer->end();
    }

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &commandBuffer->vkCommandBuffer();

    {
        MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.vkQueueSubmitBatch");
        VkResult submitResult;
        if (useSecondaryQueue_) {
            std::unique_lock<std::recursive_mutex> queueLock(device->secondaryQueueMutex());
            submitResult = vkQueueSubmit(device->secondaryQueue(), 1, &vkSubmitInfo, fence->vkFence());
        } else {
            std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
            submitResult = vkQueueSubmit(device->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence());
        }
        vk::requireSuccess(submitResult, "vkQueueSubmit(chunk BLAS batch)");
    }

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    buildingFences_.push_back(fence);
    buildingCommandBuffers_.push_back(commandBuffer);
    buildingBatches_.push_back(chunkBuildDataBatch);
}

void ChunkBuildScheduler::tryCheckBatchesFinish() {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.tryCheckBatchesFinish");
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterCommandBuffer = buildingCommandBuffers_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterCommandBuffer != buildingCommandBuffers_.end() &&
           iterBatch != buildingBatches_.end();) {
        VkResult waitResult = vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), VK_TRUE, 0);
        if (waitResult == VK_SUCCESS) {
            vk::requireSuccess(vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence()),
                               "vkResetFences(completed chunk BLAS batch)");
            (*iterCommandBuffer)->reset();
            freeFences_.push(*iterFence);
            freeCommandBuffers_.push(*iterCommandBuffer);

            publishBatchLocked(*iterBatch);

            iterFence = buildingFences_.erase(iterFence);
            iterCommandBuffer = buildingCommandBuffers_.erase(iterCommandBuffer);
            iterBatch = buildingBatches_.erase(iterBatch);
        } else if (waitResult == VK_TIMEOUT || waitResult == VK_NOT_READY) {
            ++iterFence;
            ++iterCommandBuffer;
            ++iterBatch;
        } else {
            vk::requireSuccess(waitResult, "vkWaitForFences(poll chunk BLAS batch)");
        }
    }
}

void ChunkBuildScheduler::waitAllBatchesFinish() {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.waitAllBatchesFinish");
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    uint32_t cpuSubmitBudget = std::numeric_limits<uint32_t>::max();
    submitReadyCpuBuildBatches(cpuSubmitBudget, true);

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterCommandBuffer = buildingCommandBuffers_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterCommandBuffer != buildingCommandBuffers_.end() &&
           iterBatch != buildingBatches_.end();) {
        vk::waitForFence(device->vkDevice(), (*iterFence)->vkFence(), "vkWaitForFences(chunk BLAS batch)");
        vk::requireSuccess(vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence()),
                           "vkResetFences(chunk BLAS batch)");
        (*iterCommandBuffer)->reset();
        freeFences_.push(*iterFence);
        freeCommandBuffers_.push(*iterCommandBuffer);

        publishBatchLocked(*iterBatch);

        iterFence = buildingFences_.erase(iterFence);
        iterCommandBuffer = buildingCommandBuffers_.erase(iterCommandBuffer);
        iterBatch = buildingBatches_.erase(iterBatch);
    }

    for (const auto &chunkBuildData : pendingBatchData_) {
        if (chunkBuildData == nullptr) { continue; }
        queuedIndex_.insert(chunkBuildData->id);
        chunkBuildDatas_[chunkBuildData->id] = chunkBuildData;
    }
    pendingBatchData_.clear();
    pendingBatchFrames_ = 0;
}

void ChunkBuildScheduler::tryScheduleBatches(uint32_t maxBatchSize) {
    MCVR_PROFILE_SCOPE("native.ChunkBuildScheduler.tryScheduleBatches");
    auto framework = Renderer::instance().framework();
    if (!framework->isRunning()) return;

    uint32_t readySubmitBudget = chunkBuildingTotalBatches_;
    submitReadyCpuBuildBatches(readySubmitBudget, false);

    const auto memoryBudget = framework->vma()->deviceLocalMemoryBudget();
    if (memoryBudget.valid && memoryBudget.pressure() >= 0.90f) {
        MCVR_PROFILE_METRIC("chunk_scheduler.memory_budget_throttled", 1);
        return;
    }

    uint32_t scheduledBatchCount = 0;
    while (scheduledBatchCount < chunkBuildingTotalBatches_) {
        std::shared_ptr<vk::Fence> fence;
        std::shared_ptr<vk::CommandBuffer> commandBuffer;
        std::shared_ptr<ChunkBuildDataBatch> chunkBuildDataBatch;

        {
            std::unique_lock<std::recursive_mutex> lock(mutex_);
            pruneStalePendingBatchDataLocked();

            if (freeFences_.empty() || freeCommandBuffers_.empty()) { return; }

            if (pendingBatchData_.size() < maxBatchSize && !queuedIndex_.empty()) {
                const uint32_t remainingCapacity = maxBatchSize - static_cast<uint32_t>(pendingBatchData_.size());
                glm::vec3 cameraPos = Renderer::instance().world()->getCameraPos();
                auto supplementBatch =
                    ChunkBuildDataBatch::create(remainingCapacity, queuedIndex_, chunks_, chunkBuildDatas_, cameraPos);
                pendingBatchData_.insert(pendingBatchData_.end(), supplementBatch->batchData.begin(),
                                         supplementBatch->batchData.end());
            }

            if (pendingBatchData_.empty()) { return; }

            const bool isFullBatch = pendingBatchData_.size() >= maxBatchSize;
            if (!isFullBatch) {
                pendingBatchFrames_ = pendingBatchFrames_ == 0 ? 1 : pendingBatchFrames_ + 1;
                if (pendingBatchFrames_ < maxPendingBatchFrames_) { return; }
            } else {
                pendingBatchFrames_ = 0;
            }

            fence = freeFences_.front();
            freeFences_.pop();
            commandBuffer = freeCommandBuffers_.front();
            freeCommandBuffers_.pop();

            chunkBuildDataBatch = ChunkBuildDataBatch::create(std::move(pendingBatchData_));
            pendingBatchData_.clear();
            pendingBatchFrames_ = 0;
        }

        enqueueCpuBuildJob(CpuBuildJob{
            .fence = std::move(fence),
            .commandBuffer = std::move(commandBuffer),
            .chunkBuildDataBatch = std::move(chunkBuildDataBatch),
        });
        scheduledBatchCount++;
    }
}

uint32_t ChunkBuildScheduler::chunkBuildingBatchSize() {
    return chunkBuildingBatchSize_;
}

uint32_t ChunkBuildScheduler::chunkBuildingTotalBatches() {
    return chunkBuildingTotalBatches_;
}

float Chunk1::buildFactor(std::chrono::steady_clock::time_point currentTime, glm::vec3 cameraPos, glm::vec3 chunkPos) {
    double tDiff = std::chrono::duration<double, std::milli>(currentTime - lastUpdate).count();
    double dDiff = glm::distance(cameraPos, chunkPos);

    double tScore = 1 - exp(-tDiff / T_HALF);
    double nearScore = 1 / (1 + pow(dDiff / D_HALF, D_SENSITIVITY));
    double dScore = D_PRIORITY_FLOOR + (1.0 - D_PRIORITY_FLOOR) * nearScore;
    double score = pow(tScore, T_WEIGHT) * pow(dScore, D_WEIGHT);

    return score;
}

bool Chunk1::enqueue(std::shared_ptr<ChunkBuildData> chunkBuildData) {
    MCVR_PROFILE_SCOPE("native.Chunk.enqueue");
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    lastUpdate = std::chrono::steady_clock::now();

    if (chunkBuildData->version > blasVersion && chunkBuildData->generation == generation &&
        chunkBuildData->x == x && chunkBuildData->y == y && chunkBuildData->z == z) {
        blasVersion = chunkBuildData->version;
        terrain = chunkBuildData->terrain;
        x = chunkBuildData->x;
        y = chunkBuildData->y;
        z = chunkBuildData->z;

        frr.retain(blas);
        blas = chunkBuildData->blas;

        frr.retain(indexBufferAddresses);
        indexBufferAddresses =
            std::make_shared<std::vector<VkDeviceAddress>>(std::move(chunkBuildData->indexBufferAddresses));

        frr.retain(pbrBufferAddresses);
        pbrBufferAddresses =
            std::make_shared<std::vector<VkDeviceAddress>>(std::move(chunkBuildData->pbrBufferAddresses));

        releaseBufferToPool(activeIndexBufferPool, indexBuffer, reusableFrameSerialFromNow());
        indexBuffer = chunkBuildData->indexBuffer;
        activeIndexBufferPool = chunkBuildData->indexBufferPool;

        releaseBufferToPool(activePbrBufferPool, pbrBuffer, reusableFrameSerialFromNow());
        pbrBuffer = chunkBuildData->pbrBuffer;
        activePbrBufferPool = chunkBuildData->pbrBufferPool;

        releaseBufferToPool(activeBlasStoragePool, blasStorageBuffer, reusableFrameSerialFromNow());
        blasStorageBuffer = chunkBuildData->blasStorageBuffer;
        activeBlasStoragePool = chunkBuildData->blasStoragePool;

        frr.retain(lightInfos);
        lightInfos = std::make_shared<std::vector<LightInfo>>(std::move(chunkBuildData->lightInfos));

        frr.retain(lightBuffer);
        lightBuffer = chunkBuildData->lightBuffer;
        lightCount = chunkBuildData->lightCount;

        geometryCount = chunkBuildData->geometryCount;

        frr.retain(geometryGroupNames);
        geometryGroupNames = std::make_shared<std::vector<std::string>>(std::move(chunkBuildData->geometryGroupNames));

        frr.retain(vertexCounts);
        vertexCounts = std::make_shared<std::vector<uint32_t>>();
        vertexCounts->reserve(geometryCount);

        frr.retain(indexCounts);
        indexCounts = std::make_shared<std::vector<uint32_t>>();
        indexCounts->reserve(geometryCount);
        for (uint32_t i = 0; i < geometryCount; i++) {
            vertexCounts->push_back(chunkBuildData->geometries[i].vertexCount);
            indexCounts->push_back(chunkBuildData->geometries[i].indexCount);
        }

        frr.retain(stableHitGroupNames);
        stableHitGroupNames = nullptr;
        hitGroupNameRefs.clear();
        if (blas != nullptr) {
            stableHitGroupNames = std::make_shared<std::vector<std::string>>();
            stableHitGroupNames->reserve(static_cast<size_t>(geometryCount) + 1);
            stableHitGroupNames->emplace_back("shadow");
            for (int i = 0; i < geometryCount; i++) {
                stableHitGroupNames->push_back(geometryGroupNames != nullptr &&
                                                       i < static_cast<int>(geometryGroupNames->size()) ?
                                                   (*geometryGroupNames)[i] :
                                                   "default");
            }
            hitGroupNameRefs.reserve(stableHitGroupNames->size());
            for (const std::string &groupName : *stableHitGroupNames) { hitGroupNameRefs.push_back(&groupName); }
        }
        rebuildRenderDataSnapshot();
        return true;
    } else {
        frr.retain(chunkBuildData->blas);
        frr.retain(chunkBuildData->lightBuffer);
        return false;
    }
}

void Chunk1::invalidate() {
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    lastUpdate = std::chrono::steady_clock::now();

    blasVersion = latestVersion++;
    terrain.reset();

    frr.retain(blas);
    blas = nullptr;

    frr.retain(indexBufferAddresses);
    indexBufferAddresses = nullptr;

    frr.retain(pbrBufferAddresses);
    pbrBufferAddresses = nullptr;

    releaseBufferToPool(activeIndexBufferPool, indexBuffer, reusableFrameSerialFromNow());
    indexBuffer = nullptr;
    activeIndexBufferPool = nullptr;

    releaseBufferToPool(activePbrBufferPool, pbrBuffer, reusableFrameSerialFromNow());
    pbrBuffer = nullptr;
    activePbrBufferPool = nullptr;

    releaseBufferToPool(activeBlasStoragePool, blasStorageBuffer, reusableFrameSerialFromNow());
    blasStorageBuffer = nullptr;
    activeBlasStoragePool = nullptr;

    frr.retain(lightInfos);
    lightInfos = nullptr;

    frr.retain(lightBuffer);
    lightBuffer = nullptr;
    lightCount = 0;

    geometryCount = 0;

    frr.retain(geometryGroupNames);
    geometryGroupNames = nullptr;

    frr.retain(vertexCounts);
    vertexCounts = nullptr;

    frr.retain(indexCounts);
    indexCounts = nullptr;

    frr.retain(stableHitGroupNames);
    stableHitGroupNames = nullptr;
    hitGroupNameRefs.clear();

    renderDataSnapshot_.reset();
}

void Chunk1::retainResources(FrameResourceRetainer &frr) {
    frr.retainMany(blas, indexBufferAddresses, pbrBufferAddresses, indexBuffer, pbrBuffer, blasStorageBuffer,
                   lightInfos, lightBuffer, geometryGroupNames, vertexCounts, indexCounts, stableHitGroupNames);
}

void Chunk1::releaseEmissionResources(FrameResourceRetainer &frr) {
    frr.retain(lightInfos);
    lightInfos = nullptr;

    frr.retain(lightBuffer);
    lightBuffer = nullptr;

    lightCount = 0;
    if (blas != nullptr) {
        rebuildRenderDataSnapshot();
    } else {
        renderDataSnapshot_.reset();
    }
}

void Chunk1::rebuildRenderDataSnapshot() {
    auto ret = ChunkRenderData::create();
    ret->sourceChunk = this;
    ret->x = x;
    ret->y = y;
    ret->z = z;
    ret->latestVersion = latestVersion;
    ret->blasVersion = blasVersion;
    ret->blas = blas;
    ret->indexBufferAddresses = indexBufferAddresses;
    ret->pbrBufferAddresses = pbrBufferAddresses;
    ret->indexBuffer = indexBuffer;
    ret->pbrBuffer = pbrBuffer;
    ret->blasStorageBuffer = blasStorageBuffer;
    ret->blasScratchBuffer = blasScratchBuffer;
    ret->lightInfos = lightInfos;
    ret->lightBuffer = lightBuffer;
    ret->lightCount = lightCount;
    ret->geometryCount = geometryCount;
    ret->geometryGroupNames = geometryGroupNames;
    ret->vertexCounts = vertexCounts;
    ret->indexCounts = indexCounts;

    renderDataSnapshot_ = std::move(ret);
}

std::shared_ptr<ChunkRenderData> Chunk1::tryGetValid() {
    if (renderDataSnapshot_ == nullptr) { rebuildRenderDataSnapshot(); }
    return renderDataSnapshot_;
}

const std::shared_ptr<ChunkRenderData> &Chunk1::renderDataSnapshot() const {
    return renderDataSnapshot_;
}

Chunks::Chunks(std::shared_ptr<Framework> framework) {
    MCVR_PROFILE_SCOPE("native.Chunks.init");
    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
}

void Chunks::allocateChunkPackedDataBuffers() {
    MCVR_PROFILE_SCOPE("native.Chunks.allocateChunkPackedDataBuffers");
    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    auto vma = framework->vma();

    chunkPackedDataBuffers_.clear();
    chunkPackedDataBuffers_.resize(framework->swapchain()->imageCount());
    if (chunkPackedData_.empty()) { return; }

    const size_t chunkPackedDataBytes = chunkPackedData_.size() * sizeof(ChunkPackedData);
    for (auto &chunkPackedDataBuffer : chunkPackedDataBuffers_) {
        chunkPackedDataBuffer =
            vk::DeviceLocalBuffer::create(vma, device, false, chunkPackedDataBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        chunkPackedDataBuffer->setMemoryCategory(vk::ResourceMemoryCategory::Chunk);
    }
    chunkPackedDataDirty_ = true;
}

void Chunks::clearEntityLightDataLocked() {
    if (!chunkPackedData_.empty()) {
        chunkPackedData_[0].globalEntityLightCount = 0;
        chunkPackedData_[0].globalEntityLightBufferAddress = 0;
    }
    if (entityLightDataChunkIndices_.empty()) { return; }

    for (uint32_t chunkIndex : entityLightDataChunkIndices_) {
        if (chunkIndex >= chunkPackedData_.size()) { continue; }
        ChunkPackedData &data = chunkPackedData_[chunkIndex];
        data.entityLightCount = 0;
        data.entityLightBufferAddress = 0;
    }
    entityLightDataChunkIndices_.clear();
}

void Chunks::uploadChunkPackedDataForCurrentFrameLocked() {
    if (!Renderer::options.collectChunkEmission || chunkPackedData_.empty() || !chunkPackedDataDirty_) { return; }

    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    if (chunkPackedDataBuffers_.empty()) { allocateChunkPackedDataBuffers(); }
    if (context->frameIndex >= chunkPackedDataBuffers_.size()) { return; }

    auto &chunkPackedDataBuffer = chunkPackedDataBuffers_[context->frameIndex];
    if (chunkPackedDataBuffer == nullptr) { return; }

    const size_t chunkPackedDataBytes = chunkPackedData_.size() * sizeof(ChunkPackedData);
    MCVR_PROFILE_METRIC("chunk_packed_data.bytes", chunkPackedDataBytes);
    chunkPackedDataBuffer->uploadToStagingBuffer(chunkPackedData_.data(), chunkPackedDataBytes, 0);
    Renderer::instance().buffers()->queueImportantWorldUpload(chunkPackedDataBuffer, chunkPackedDataBytes, 0);
    chunkPackedDataDirty_ = false;
}

void Chunks::releaseEmissionResources() {
    MCVR_PROFILE_SCOPE("native.Chunks.releaseEmissionResources");
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    for (auto &chunk : chunks_) {
        if (chunk != nullptr) { chunk->releaseEmissionResources(frr); }
    }

    for (auto &chunkBuildData : chunkBuildDatas_) {
        if (chunkBuildData == nullptr) { continue; }
        chunkBuildData->lightInfos.clear();
        frr.retain(chunkBuildData->lightBuffer);
        chunkBuildData->lightBuffer = nullptr;
        chunkBuildData->lightCount = 0;
    }

    rebuildRenderableChunkSnapshotsLocked();

    entityLightDataChunkIndices_.clear();
    std::fill(chunkPackedData_.begin(), chunkPackedData_.end(), ChunkPackedData{});
    chunkPackedDataDirty_ = true;
    for (auto &chunkPackedDataBuffer : chunkPackedDataBuffers_) {
        frr.retain(chunkPackedDataBuffer);
        chunkPackedDataBuffer = nullptr;
    }
    chunkPackedDataBuffers_.clear();
}

bool Chunks::isBuildDataCurrentLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData) const {
    if (chunkBuildData == nullptr) { return false; }

    const int64_t chunkId = chunkBuildData->id;
    if (chunkId < 0 || static_cast<size_t>(chunkId) >= chunks_.size()) { return false; }

    const auto &chunk = chunks_[chunkId];
    if (chunk == nullptr) { return false; }

    return chunkBuildData->version + 1 == chunk->latestVersion &&
           chunkBuildData->generation == chunk->generation && chunkBuildData->x == chunk->x &&
           chunkBuildData->y == chunk->y && chunkBuildData->z == chunk->z;
}

void Chunks::queueImportantBuildUploadsLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData) {
    MCVR_PROFILE_SCOPE("native.Chunks.queueImportantBuildUploadsLocked");
    if (chunkBuildData->pbrBuffer != nullptr) {
        Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->indexBuffer);
        Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->pbrBuffer);
    }
    if (chunkBuildData->lightBuffer != nullptr) {
        Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->lightBuffer);
    }
    if (chunkBuildData->blasBuilder != nullptr) { importantBLASBuilders_->push_back(chunkBuildData->blasBuilder); }
}

void Chunks::publishImportantBuildDataLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData) {
    MCVR_PROFILE_SCOPE("native.Chunks.publishImportantBuildDataLocked");
    if (!isBuildDataCurrentLocked(chunkBuildData)) {
        releaseChunkBuildDataBuffersToPools(chunkBuildData, true);
        return;
    }

    queueImportantBuildUploadsLocked(chunkBuildData);
    if (!chunks_[chunkBuildData->id]->enqueue(chunkBuildData)) {
        releaseChunkBuildDataBuffersToPools(chunkBuildData, true);
        return;
    }
    releaseChunkBuildDataBuffersToPools(chunkBuildData, false);

    auto slotUpdateInfo =
        updateRenderableChunkSlot(renderableChunks_, renderableChunkSnapshots_, renderableChunkIds_,
                                  renderableChunkSlots_, chunkBuildData->id, chunks_[chunkBuildData->id]);
    applyRenderableChunkSlotUpdate(chunkGeometryMetadataCache_, chunks_, slotUpdateInfo, chunkBuildData->id,
                                   renderableChunks_.size());
    storeChunkPackedData(chunkPackedData_, chunkBuildData->id, chunkBuildData->x, chunkBuildData->y, chunkBuildData->z,
                         chunkBuildData->geometryCount, chunkBuildData->lightCount,
                         chunkBuildData->lightBuffer != nullptr ? chunkBuildData->lightBuffer->bufferAddress() : 0);
}

void Chunks::rebuildRenderableChunkSnapshotsLocked() {
    MCVR_PROFILE_SCOPE("native.Chunks.rebuildRenderableChunkSnapshotsLocked");
    renderableChunkSnapshots_.clear();
    renderableChunkSnapshots_.reserve(renderableChunks_.size());
    for (const auto &chunk : renderableChunks_) {
        renderableChunkSnapshots_.push_back(chunk == nullptr ? nullptr : chunk->renderDataSnapshot());
    }
}

void Chunks::reset(uint32_t numChunks, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, int32_t bottomSectionCoord) {
    MCVR_PROFILE_SCOPE("native.Chunks.reset");
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    {
        std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
        vk::requireSuccess(vkQueueWaitIdle(device->mainVkQueue()), "vkQueueWaitIdle(reset chunks, main queue)");
    }
    if (device->hasIndependentSecondaryQueue()) {
        std::unique_lock<std::recursive_mutex> queueLock(device->secondaryQueueMutex());
        vk::requireSuccess(vkQueueWaitIdle(device->secondaryQueue()), "vkQueueWaitIdle(reset chunks, backend queue)");
    }

    sizeX_ = static_cast<int32_t>(sizeX);
    sizeY_ = static_cast<int32_t>(sizeY);
    sizeZ_ = static_cast<int32_t>(sizeZ);
    bottomSectionCoord_ = bottomSectionCoord;
    chunkStorageSectionPos_ = glm::ivec3(0, bottomSectionCoord, 0);

    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
    importantFrameBufferPools_.clear();
    activeChunkBufferPools_.clear();

    chunks_.clear();
    chunks_.resize(numChunks);
    renderableChunks_.clear();
    renderableChunkSnapshots_.clear();
    renderableChunkIds_.clear();
    renderableChunkSlots_.assign(numChunks, unusedChunkSlot);
    chunkGeometryMetadataCache_.reset(numChunks);
    chunkPackedData_.assign(numChunks, ChunkPackedData{});
    chunkPackedDataBuffers_.clear();
    entityLightDataChunkIndices_.clear();
    chunkPackedDataDirty_ = true;
    if (Renderer::options.collectChunkEmission) { allocateChunkPackedDataBuffers(); }
    chunkBuildDatas_.clear();
    chunkBuildDatas_.resize(numChunks);
    queuedIndex_.clear();

    for (int i = 0; i < numChunks; i++) {
        chunks_[i] = Chunk1::create();
        chunks_[i]->indexBufferPool = ChunkGpuBufferPool::create();
        chunks_[i]->pbrBufferPool = ChunkGpuBufferPool::create();
        chunks_[i]->blasStoragePool = ChunkGpuBufferPool::create();
        chunks_[i]->blasScratchPool = ChunkGpuBufferPool::create();
        chunkBuildDatas_[i] = nullptr;
    }

    chunkBuildScheduler_ = ChunkBuildScheduler::create(
        queuedIndex_, chunks_, chunkBuildDatas_, mutex_, chunkPackedData_, renderableChunks_, renderableChunkSnapshots_,
        renderableChunkIds_, renderableChunkSlots_, chunkGeometryMetadataCache_,
        Renderer::instance().options.chunkBuildingBatchSize, Renderer::instance().options.chunkBuildingTotalBatches,
        Renderer::instance().options.chunkBuildingThreads);
}

void Chunks::resetScheduler() {
    MCVR_PROFILE_SCOPE("native.Chunks.resetScheduler");
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (chunkBuildScheduler_ == nullptr) return;

    chunkBuildScheduler_->waitAllBatchesFinish();

    chunkBuildScheduler_ = ChunkBuildScheduler::create(
        queuedIndex_, chunks_, chunkBuildDatas_, mutex_, chunkPackedData_, renderableChunks_, renderableChunkSnapshots_,
        renderableChunkIds_, renderableChunkSlots_, chunkGeometryMetadataCache_,
        Renderer::instance().options.chunkBuildingBatchSize, Renderer::instance().options.chunkBuildingTotalBatches,
        Renderer::instance().options.chunkBuildingThreads);
}

void Chunks::ensureImportantFrameBufferPools() {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr || framework->swapchain() == nullptr) {
        throw std::runtime_error("important chunk frame pools require an initialized swapchain");
    }

    const size_t frameCount = framework->swapchain()->imageCount();
    if (importantFrameBufferPools_.size() < frameCount) { importantFrameBufferPools_.resize(frameCount); }
    for (auto &pools : importantFrameBufferPools_) {
        if (pools.indexBufferPool == nullptr) { pools.indexBufferPool = ChunkGpuBufferPool::create(); }
        if (pools.pbrBufferPool == nullptr) { pools.pbrBufferPool = ChunkGpuBufferPool::create(); }
        if (pools.blasStoragePool == nullptr) { pools.blasStoragePool = ChunkGpuBufferPool::create(); }
        if (pools.blasScratchPool == nullptr) { pools.blasScratchPool = ChunkGpuBufferPool::create(); }
    }
}

Chunks::ChunkFrameBufferPools &Chunks::importantFrameBufferPoolsForCurrentFrame() {
    ensureImportantFrameBufferPools();
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    if (context == nullptr || context->frameIndex >= importantFrameBufferPools_.size()) {
        throw std::runtime_error("important chunk frame pool index is out of range");
    }
    return importantFrameBufferPools_[context->frameIndex];
}

void Chunks::setCollectChunkEmission(bool collect) {
    MCVR_PROFILE_SCOPE("native.Chunks.setCollectChunkEmission");
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (chunkBuildScheduler_ != nullptr) { chunkBuildScheduler_->waitAllBatchesFinish(); }

    auto textures = Renderer::instance().textures();
    auto emission = textures != nullptr ? textures->emission() : nullptr;
    for (auto &chunkBuildData : chunkBuildDatas_) {
        if (chunkBuildData == nullptr) { continue; }

        chunkBuildData->collectChunkEmission = collect;
        if (collect && emission != nullptr) {
            chunkBuildData->buildLightInfos(*emission);
        } else {
            chunkBuildData->lightInfos.clear();
            chunkBuildData->lightCount = 0;
            chunkBuildData->lightBuffer = nullptr;
        }
    }

    if (collect) {
        if (chunkPackedDataBuffers_.empty()) { allocateChunkPackedDataBuffers(); }
        return;
    }

    releaseEmissionResources();
}

void Chunks::trackChunkBufferPoolForSweep(const std::shared_ptr<ChunkGpuBufferPool> &pool) {
    if (pool == nullptr || !pool->markSweepQueued()) { return; }
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    activeChunkBufferPools_.push_back(pool);
}

static VkDeviceSize chunkBufferSize(const std::shared_ptr<vk::DeviceLocalBuffer> &buffer) {
    return buffer != nullptr ? buffer->size() : 0;
}

static void
addPoolStats(const std::shared_ptr<ChunkGpuBufferPool> &pool, VkDeviceSize &capacityBytes, size_t &entryCount) {
    if (pool == nullptr) { return; }
    const ChunkGpuBufferPoolStats stats = pool->stats();
    capacityBytes += stats.capacityBytes;
    entryCount += stats.entryCount;
}

void Chunks::recordChunkMemoryBreakdownLocked() const {
#if defined(MCVR_ENABLE_PROFILER)
    if ((frameSerial_ % 30) != 0) { return; }

    VkDeviceSize activePbrCapacityBytes = 0;
    VkDeviceSize activeIndexCapacityBytes = 0;
    VkDeviceSize activeBlasStorageCapacityBytes = 0;
    VkDeviceSize activeLightCapacityBytes = 0;
    VkDeviceSize activePbrUsedBytes = 0;
    VkDeviceSize activeIndexUsedBytes = 0;
    uint64_t activeChunkCount = 0;
    uint64_t activeGeometryCount = 0;
    uint64_t activeVertexCount = 0;
    uint64_t activeIndexCount = 0;

    VkDeviceSize idleIndexCapacityBytes = 0;
    VkDeviceSize idlePbrCapacityBytes = 0;
    VkDeviceSize idleBlasStorageCapacityBytes = 0;
    VkDeviceSize idleBlasScratchCapacityBytes = 0;
    size_t idleIndexEntryCount = 0;
    size_t idlePbrEntryCount = 0;
    size_t idleBlasStorageEntryCount = 0;
    size_t idleBlasScratchEntryCount = 0;

    for (const auto &chunk : chunks_) {
        if (chunk == nullptr) { continue; }

        const bool hasActiveBuffer = chunk->indexBuffer != nullptr || chunk->pbrBuffer != nullptr ||
                                     chunk->blasStorageBuffer != nullptr || chunk->lightBuffer != nullptr;
        if (hasActiveBuffer) {
            activeChunkCount++;
            activePbrCapacityBytes += chunkBufferSize(chunk->pbrBuffer);
            activeIndexCapacityBytes += chunkBufferSize(chunk->indexBuffer);
            activeBlasStorageCapacityBytes += chunkBufferSize(chunk->blasStorageBuffer);
            activeLightCapacityBytes += chunkBufferSize(chunk->lightBuffer);
            activeGeometryCount += chunk->geometryCount;

            if (chunk->vertexCounts != nullptr) {
                for (uint32_t vertexCount : *chunk->vertexCounts) { activeVertexCount += vertexCount; }
            }
            if (chunk->indexCounts != nullptr) {
                for (uint32_t indexCount : *chunk->indexCounts) { activeIndexCount += indexCount; }
            }
        }

        addPoolStats(chunk->indexBufferPool, idleIndexCapacityBytes, idleIndexEntryCount);
        addPoolStats(chunk->pbrBufferPool, idlePbrCapacityBytes, idlePbrEntryCount);
        addPoolStats(chunk->blasStoragePool, idleBlasStorageCapacityBytes, idleBlasStorageEntryCount);
        addPoolStats(chunk->blasScratchPool, idleBlasScratchCapacityBytes, idleBlasScratchEntryCount);
    }

    activePbrUsedBytes = activeVertexCount * sizeof(vk::VertexFormat::ChunkPBRVertex);
    activeIndexUsedBytes = activeIndexCount * sizeof(uint32_t);

    VkDeviceSize importantIndexCapacityBytes = 0;
    VkDeviceSize importantPbrCapacityBytes = 0;
    VkDeviceSize importantBlasStorageCapacityBytes = 0;
    VkDeviceSize importantBlasScratchCapacityBytes = 0;
    size_t importantIndexEntryCount = 0;
    size_t importantPbrEntryCount = 0;
    size_t importantBlasStorageEntryCount = 0;
    size_t importantBlasScratchEntryCount = 0;
    for (const auto &pools : importantFrameBufferPools_) {
        addPoolStats(pools.indexBufferPool, importantIndexCapacityBytes, importantIndexEntryCount);
        addPoolStats(pools.pbrBufferPool, importantPbrCapacityBytes, importantPbrEntryCount);
        addPoolStats(pools.blasStoragePool, importantBlasStorageCapacityBytes, importantBlasStorageEntryCount);
        addPoolStats(pools.blasScratchPool, importantBlasScratchCapacityBytes, importantBlasScratchEntryCount);
    }

    VkDeviceSize metadataCapacityBytes = chunkBufferSize(chunkGeometryMetadataCache_.renderRecordBuffer) +
                                         chunkBufferSize(chunkGeometryMetadataCache_.indexBufferAddressBuffer) +
                                         chunkBufferSize(chunkGeometryMetadataCache_.pbrBufferAddressBuffer);
    VkDeviceSize packedDataCapacityBytes = 0;
    for (const auto &buffer : chunkPackedDataBuffers_) { packedDataCapacityBytes += chunkBufferSize(buffer); }

    const VkDeviceSize activeGeometryCapacityBytes =
        activePbrCapacityBytes + activeIndexCapacityBytes + activeBlasStorageCapacityBytes;
    const VkDeviceSize idlePoolCapacityBytes =
        idleIndexCapacityBytes + idlePbrCapacityBytes + idleBlasStorageCapacityBytes + idleBlasScratchCapacityBytes;
    const VkDeviceSize importantPoolCapacityBytes = importantIndexCapacityBytes + importantPbrCapacityBytes +
                                                    importantBlasStorageCapacityBytes +
                                                    importantBlasScratchCapacityBytes;
    const VkDeviceSize knownChunkDeviceBytes = activeGeometryCapacityBytes + activeLightCapacityBytes +
                                               idlePoolCapacityBytes + importantPoolCapacityBytes +
                                               metadataCapacityBytes + packedDataCapacityBytes;

    MCVR_PROFILE_SCOPE("native.Chunks.recordChunkMemoryBreakdown");
    MCVR_PROFILE_METRIC("chunk_memory.sample_frame_serial", frameSerial_);
    MCVR_PROFILE_METRIC("chunk_memory.active.chunk_count", activeChunkCount);
    MCVR_PROFILE_METRIC("chunk_memory.active.renderable_chunk_count", renderableChunks_.size());
    MCVR_PROFILE_METRIC("chunk_memory.active.geometry_count", activeGeometryCount);
    MCVR_PROFILE_METRIC("chunk_memory.active.vertex_count", activeVertexCount);
    MCVR_PROFILE_METRIC("chunk_memory.active.index_count", activeIndexCount);
    MCVR_PROFILE_METRIC("chunk_memory.active.pbr_capacity_bytes", activePbrCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.active.pbr_used_bytes", activePbrUsedBytes);
    MCVR_PROFILE_METRIC("chunk_memory.active.pbr_slack_bytes",
                        activePbrCapacityBytes > activePbrUsedBytes ? activePbrCapacityBytes - activePbrUsedBytes : 0);
    MCVR_PROFILE_METRIC("chunk_memory.active.index_capacity_bytes", activeIndexCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.active.index_used_bytes", activeIndexUsedBytes);
    MCVR_PROFILE_METRIC("chunk_memory.active.index_slack_bytes", activeIndexCapacityBytes > activeIndexUsedBytes ?
                                                                     activeIndexCapacityBytes - activeIndexUsedBytes :
                                                                     0);
    MCVR_PROFILE_METRIC("chunk_memory.active.blas_storage_capacity_bytes", activeBlasStorageCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.active.light_capacity_bytes", activeLightCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.active.geometry_capacity_bytes", activeGeometryCapacityBytes);

    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.index_capacity_bytes", idleIndexCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.pbr_capacity_bytes", idlePbrCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.blas_storage_capacity_bytes", idleBlasStorageCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.blas_scratch_capacity_bytes", idleBlasScratchCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.total_capacity_bytes", idlePoolCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.index_entry_count", idleIndexEntryCount);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.pbr_entry_count", idlePbrEntryCount);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.blas_storage_entry_count", idleBlasStorageEntryCount);
    MCVR_PROFILE_METRIC("chunk_memory.idle_pool.blas_scratch_entry_count", idleBlasScratchEntryCount);

    MCVR_PROFILE_METRIC("chunk_memory.important_pool.index_capacity_bytes", importantIndexCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.pbr_capacity_bytes", importantPbrCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.blas_storage_capacity_bytes", importantBlasStorageCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.blas_scratch_capacity_bytes", importantBlasScratchCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.total_capacity_bytes", importantPoolCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.index_entry_count", importantIndexEntryCount);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.pbr_entry_count", importantPbrEntryCount);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.blas_storage_entry_count", importantBlasStorageEntryCount);
    MCVR_PROFILE_METRIC("chunk_memory.important_pool.blas_scratch_entry_count", importantBlasScratchEntryCount);

    MCVR_PROFILE_METRIC("chunk_memory.metadata.render_record_count", chunkGeometryMetadataCache_.renderRecords.size());
    MCVR_PROFILE_METRIC("chunk_memory.metadata.geometry_slot_count", chunkGeometryMetadataCache_.geometrySlotCount);
    MCVR_PROFILE_METRIC("chunk_memory.metadata.actual_geometry_count", chunkGeometryMetadataCache_.actualGeometryCount);
    MCVR_PROFILE_METRIC("chunk_memory.metadata.buffer_capacity_bytes", metadataCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.packed_data.buffer_capacity_bytes", packedDataCapacityBytes);
    MCVR_PROFILE_METRIC("chunk_memory.known_device_bytes", knownChunkDeviceBytes);
#endif
}

void Chunks::resetFrame() {
    MCVR_PROFILE_SCOPE("native.Chunks.resetFrame");
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    frameSerial_++;
    const auto memoryBudget = framework->vma()->deviceLocalMemoryBudget();
    const bool memoryPressure = memoryBudget.valid && memoryBudget.pressure() >= 0.85f;
    if (memoryPressure != memoryPressureActive_) {
        if (memoryPressure) {
            std::cerr << "[Chunks] Device-local memory pressure reached "
                      << static_cast<uint32_t>(memoryBudget.pressure() * 100.0f) << "% ("
                      << memoryBudget.usage / (1024ull * 1024ull) << " MiB used, "
                      << memoryBudget.budget / (1024ull * 1024ull) << " MiB budget); trimming reusable chunk buffers."
                      << std::endl;
        }
        memoryPressureActive_ = memoryPressure;
    }
    MCVR_PROFILE_METRIC("chunk_pool.device_local_budget_bytes", memoryBudget.budget);
    MCVR_PROFILE_METRIC("chunk_pool.device_local_usage_bytes", memoryBudget.usage);
    MCVR_PROFILE_METRIC("chunk_pool.memory_pressure", memoryPressure ? 1 : 0);
    ensureImportantFrameBufferPools();
    {
        MCVR_PROFILE_SCOPE("native.Chunks.resetFrame.sweepImportantFramePools");
        MCVR_PROFILE_METRIC("chunk_pool.important_frame_pool_count", importantFrameBufferPools_.size());
        for (auto &pools : importantFrameBufferPools_) {
            if (pools.indexBufferPool != nullptr) { pools.indexBufferPool->sweep(frameSerial_, memoryPressure); }
            if (pools.pbrBufferPool != nullptr) { pools.pbrBufferPool->sweep(frameSerial_, memoryPressure); }
            if (pools.blasStoragePool != nullptr) { pools.blasStoragePool->sweep(frameSerial_, memoryPressure); }
            if (pools.blasScratchPool != nullptr) { pools.blasScratchPool->sweep(frameSerial_, memoryPressure); }
        }
    }
    {
        MCVR_PROFILE_SCOPE("native.Chunks.resetFrame.sweepChunkPools");
        MCVR_PROFILE_METRIC("chunk_pool.chunk_count", chunks_.size());
        MCVR_PROFILE_METRIC("chunk_pool.active_sweep_pool_count", activeChunkBufferPools_.size());
        std::vector<std::shared_ptr<ChunkGpuBufferPool>> remainingActivePools;
        remainingActivePools.reserve(activeChunkBufferPools_.size());
        for (const auto &pool : activeChunkBufferPools_) {
            if (pool == nullptr) { continue; }
            pool->sweep(frameSerial_, memoryPressure);
            if (pool->hasEntries()) {
                remainingActivePools.push_back(pool);
            } else {
                pool->clearSweepQueued();
            }
        }
        activeChunkBufferPools_.swap(remainingActivePools);
        MCVR_PROFILE_METRIC("chunk_pool.remaining_active_sweep_pool_count", activeChunkBufferPools_.size());
    }

    recordChunkMemoryBreakdownLocked();

    {
        MCVR_PROFILE_SCOPE("native.Chunks.resetFrame.resetImportantBuilders");
        frr.retain(importantBLASBuilders_);
        importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
    }

    {
        MCVR_PROFILE_SCOPE("native.Chunks.resetFrame.uploadChunkPackedData");
        clearEntityLightDataLocked();
        chunkPackedDataDirty_ = true;
    }
}

uint64_t Chunks::frameSerial() const {
    return frameSerial_;
}

uint64_t Chunks::bufferReuseDelayFrames() const {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr || framework->swapchain() == nullptr) { return 3; }
    return static_cast<uint64_t>(framework->swapchain()->imageCount()) + 1;
}

void Chunks::invalidateChunk(int id, int x, int y, int z, int64_t generation,
                             std::shared_ptr<const VistaTerrain> terrain) {
    MCVR_PROFILE_SCOPE("native.Chunks.invalidateChunk");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id < 0 || static_cast<size_t>(id) >= chunks_.size()) { return; }
    const auto &chunk = chunks_[id];
    if (chunk->generation != generation || chunk->x != x || chunk->y != y || chunk->z != z) { return; }
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    queuedIndex_.erase(id);

    frr.retain(chunkBuildDatas_[id]);
    chunkBuildDatas_[id] = nullptr;

    chunks_[id]->invalidate();
    chunks_[id]->terrainEmpty = true;
    chunks_[id]->terrain = std::move(terrain);
    auto slotUpdateInfo = updateRenderableChunkSlot(renderableChunks_, renderableChunkSnapshots_, renderableChunkIds_,
                                                    renderableChunkSlots_, id, chunks_[id]);
    applyRenderableChunkSlotUpdate(chunkGeometryMetadataCache_, chunks_, slotUpdateInfo, id, renderableChunks_.size());

    storeChunkPackedData(chunkPackedData_, id, chunks_[id]->x, chunks_[id]->y, chunks_[id]->z, 0, 0, 0);
}

void Chunks::relocateChunk(int id, int x, int y, int z, int64_t generation) {
    MCVR_PROFILE_SCOPE("native.Chunks.relocateChunk");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id < 0 || static_cast<size_t>(id) >= chunks_.size() || generation <= chunks_[id]->generation) { return; }
    chunks_[id]->generation = generation;
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    queuedIndex_.erase(id);

    frr.retain(chunkBuildDatas_[id]);
    chunkBuildDatas_[id] = nullptr;

    chunks_[id]->x = x;
    chunks_[id]->y = y;
    chunks_[id]->z = z;
    chunks_[id]->invalidate();
    chunks_[id]->terrainEmpty = false;
    auto slotUpdateInfo = updateRenderableChunkSlot(renderableChunks_, renderableChunkSnapshots_, renderableChunkIds_,
                                                    renderableChunkSlots_, id, chunks_[id]);
    applyRenderableChunkSlotUpdate(chunkGeometryMetadataCache_, chunks_, slotUpdateInfo, id, renderableChunks_.size());

    storeChunkPackedData(chunkPackedData_, id, x, y, z, 0, 0, 0);
}

void Chunks::queueChunkBuild(ChunkBuildTask task) {
    MCVR_PROFILE_SCOPE("native.Chunks.queueChunkBuild");
    MCVR_PROFILE_METRIC("chunk_queue.geometry_count", task.geometryCount);
    MCVR_PROFILE_METRIC("chunk_queue.important", task.isImportant ? 1 : 0);
    uint32_t allVertexCount = 0, allIndexCount = 0;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::string> geometryGroupNames;
    std::vector<ChunkBuildGeometry> geometries;
    geometries.reserve(task.geometryCount);

    {
        MCVR_PROFILE_SCOPE("native.Chunks.queueChunkBuild.collectGeometry");
        for (int i = 0; i < task.geometryCount; i++) {
            World::GeometryTypes geometryType = static_cast<World::GeometryTypes>(task.geometryTypes[i]);
            const uint32_t vertexCount = static_cast<uint32_t>(task.vertexCounts[i]);
            const uint32_t generatedIndexCount = static_cast<uint32_t>(task.vertexCounts[i] / 4 * 6);
            ChunkBuildGeometry::IndexSource indexSource = ChunkBuildGeometry::IndexSource::GeneratedQuads;
            const void *indexSourcePointer = nullptr;
            if (task.indices != nullptr && task.indices[i] != nullptr) {
                const int indexType = task.indexTypes != nullptr ? task.indexTypes[i] : 1;
                if (indexType == 0) {
                    indexSource = ChunkBuildGeometry::IndexSource::ExternalU16;
                } else {
                    indexSource = ChunkBuildGeometry::IndexSource::ExternalU32;
                }
                indexSourcePointer = task.indices[i];
            }

            if (task.transferredBufferLease == nullptr ||
                !task.transferredBufferLease->valid(static_cast<uint32_t>(i))) {
                throw std::runtime_error("Chunk build geometry is missing a transferred Java buffer");
            }
            if (task.vertexFormats == nullptr || task.vertexFormats[i] != World::CHUNK_PBR_TRIANGLE) {
                throw std::runtime_error("Chunk build geometry must use chunk PBR vertex format");
            }

            auto geometry = ChunkBuildGeometry::externalPBR(
                task.vertices[i], vertexCount, indexSource, indexSourcePointer, generatedIndexCount,
                task.transferredBufferLease->retain(static_cast<uint32_t>(i)));
            if (geometry.empty()) { continue; }

            geometryTypes.push_back(geometryType);
            if (task.geometryGroupNames != nullptr && task.geometryGroupNames[i] != nullptr) {
                geometryGroupNames.emplace_back(task.geometryGroupNames[i]);
            } else {
                geometryGroupNames.emplace_back("default");
            }

            allVertexCount += geometry.vertexCount;
            allIndexCount += geometry.indexCount;
            geometries.push_back(std::move(geometry));
        }
    }
    MCVR_PROFILE_METRIC("chunk_queue.collected_geometry_count", geometries.size());
    MCVR_PROFILE_METRIC("chunk_queue.vertex_count", allVertexCount);
    MCVR_PROFILE_METRIC("chunk_queue.index_count", allIndexCount);

    const bool collectChunkEmission = Renderer::options.collectChunkEmission;
    auto framework = Renderer::instance().framework();
    auto &frr = framework->frameResourceRetainer();

    int64_t chunkVersion = 0;
    {
        MCVR_PROFILE_SCOPE("native.Chunks.queueChunkBuild.version");
        std::unique_lock<std::recursive_mutex> lock(mutex_);
        if (task.id < 0 || static_cast<size_t>(task.id) >= chunks_.size()) { return; }
        const auto &chunk = chunks_[task.id];
        if (task.generation != chunk->generation || task.x != chunk->x || task.y != chunk->y || task.z != chunk->z) {
            return;
        }
        chunks_[task.id]->terrainEmpty = false;
        chunkVersion = chunks_[task.id]->latestVersion++;
        if (task.isImportant) {
            queuedIndex_.erase(task.id);
            frr.retain(chunkBuildDatas_[task.id]);
            chunkBuildDatas_[task.id] = nullptr;
        }
    }

    std::shared_ptr<ChunkBuildData> chunkBuildData =
        ChunkBuildData::create(task.id, task.x, task.y, task.z, chunkVersion, collectChunkEmission, task.isImportant,
                               allVertexCount, allIndexCount, static_cast<uint32_t>(geometries.size()),
                               std::move(geometryTypes), std::move(geometryGroupNames), std::move(geometries));
    chunkBuildData->generation = task.generation;
    chunkBuildData->terrain = std::move(task.terrain);
    {
        MCVR_PROFILE_SCOPE("native.Chunks.queueChunkBuild.bindPools");
        std::unique_lock<std::recursive_mutex> lock(mutex_);
        if (!isBuildDataCurrentLocked(chunkBuildData)) { return; }
        bindChunkBuildDataPools(chunkBuildData, chunks_[task.id]);
    }

    if (collectChunkEmission && (Renderer::instance().textures() != nullptr)) {
        auto textures = Renderer::instance().textures();
        if (auto emission = textures->emission(); emission != nullptr) { chunkBuildData->buildLightInfos(*emission); }
    }

    {
        MCVR_PROFILE_SCOPE(task.isImportant ? "native.Chunks.queueChunkBuild.enqueueImportant" :
                                              "native.Chunks.queueChunkBuild.enqueueNormal");
        std::unique_lock<std::recursive_mutex> lock(mutex_);
        if (!isBuildDataCurrentLocked(chunkBuildData)) { return; }
        queuedIndex_.insert(task.id);
        chunkBuildDatas_[task.id] = chunkBuildData;
    }
}

bool Chunks::isChunkReady(int64_t id) {
    MCVR_PROFILE_SCOPE("native.Chunks.isChunkReady");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id < 0 || static_cast<size_t>(id) >= chunks_.size()) { return false; }
    auto chunkRenderData = chunks_[id]->tryGetValid();
    return chunkRenderData->blas != nullptr;
}

void Chunks::close() {
    MCVR_PROFILE_SCOPE("native.Chunks.close");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (chunkBuildScheduler_ != nullptr) {
        chunkBuildScheduler_->waitAllBatchesFinish();
        chunkBuildScheduler_ = nullptr;
    }

    queuedIndex_.clear();
    chunkBuildDatas_.clear();
    importantBLASBuilders_ = nullptr;
    importantFrameBufferPools_.clear();
    chunkPackedData_.clear();
    chunkPackedDataBuffers_.clear();
    entityLightDataChunkIndices_.clear();
    chunkPackedDataDirty_ = false;
    renderableChunks_.clear();
    renderableChunkSnapshots_.clear();
    renderableChunkIds_.clear();
    renderableChunkSlots_.clear();
    activeChunkBufferPools_.clear();
    chunkGeometryMetadataCache_.clear();
    chunks_.clear();
    sizeX_ = 0;
    sizeY_ = 0;
    sizeZ_ = 0;
    bottomSectionCoord_ = 0;
    chunkStorageSectionPos_ = glm::ivec3(0);
}

std::recursive_mutex &Chunks::mutex() {
    return mutex_;
}

std::vector<std::shared_ptr<Chunk1>> &Chunks::chunks() {
    return chunks_;
}

const std::vector<std::shared_ptr<Chunk1>> &Chunks::renderableChunks() {
    return renderableChunks_;
}

const std::vector<std::shared_ptr<ChunkRenderData>> &Chunks::renderableChunkSnapshots() {
    return renderableChunkSnapshots_;
}

const ChunkGeometryMetadataCache &Chunks::chunkGeometryMetadataCache() {
    return chunkGeometryMetadataCache_;
}

void Chunks::clearChunkGeometryHitGroupDirtyRanges() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    chunkGeometryMetadataCache_.hitGroupDirtyRanges.clear();
}

void Chunks::clearChunkRasterDrawDirtySlots() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    chunkGeometryMetadataCache_.clearRasterDrawDirtySlots();
}

template <typename T>
static bool prepareMetadataBuffer(std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                  const std::vector<T> &data,
                                  const std::shared_ptr<vk::VMA> &vma,
                                  const std::shared_ptr<vk::Device> &device,
                                  FrameResourceRetainer &frr,
                                  VkBufferUsageFlags usage) {
    if (data.empty()) {
        if (buffer != nullptr) {
            frr.retain(buffer);
            buffer = nullptr;
        }
        return false;
    }

    const size_t byteSize = data.size() * sizeof(T);
    if (buffer == nullptr || buffer->size() < byteSize) {
        frr.retain(buffer);
        buffer = createMetadataBuffer(data, vma, device, usage);
        return buffer != nullptr;
    }
    return false;
}

template <typename T>
static bool uploadMetadataRanges(const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                 const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                                 const std::vector<T> &data,
                                 const std::vector<ChunkMetadataUpdateRangeInfo> &ranges) {
    if (buffer == nullptr || data.empty() || ranges.empty()) { return false; }

    bool uploaded = false;
    for (const auto &range : ranges) {
        if (range.count == 0 || range.offset >= data.size()) { continue; }
        const size_t count = std::min<size_t>(range.count, data.size() - range.offset);
        const size_t byteOffset = static_cast<size_t>(range.offset) * sizeof(T);
        const size_t byteSize = count * sizeof(T);
        buffer->uploadToStagingBuffer(const_cast<T *>(data.data() + range.offset), byteSize, byteOffset);
        buffer->uploadToBuffer(commandBuffer, byteSize, byteOffset, byteOffset);
        uploaded = true;
    }
    return uploaded;
}

void Chunks::uploadChunkGeometryMetadataCache(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    MCVR_PROFILE_SCOPE("native.Chunks.uploadChunkGeometryMetadataCache");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (!chunkGeometryMetadataCache_.hasPendingGpuUpload) { return; }

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto &frr = framework->frameResourceRetainer();

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> buffers;
    buffers.reserve(3);
    bool fullRenderRecordUpload =
        prepareMetadataBuffer(chunkGeometryMetadataCache_.renderRecordBuffer, chunkGeometryMetadataCache_.renderRecords,
                              vma, device, frr, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    bool fullIndexAddressUpload = prepareMetadataBuffer(chunkGeometryMetadataCache_.indexBufferAddressBuffer,
                                                        chunkGeometryMetadataCache_.indexBufferAddresses, vma, device,
                                                        frr, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    bool fullPbrAddressUpload = prepareMetadataBuffer(chunkGeometryMetadataCache_.pbrBufferAddressBuffer,
                                                      chunkGeometryMetadataCache_.pbrBufferAddresses, vma, device, frr,
                                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    if (fullRenderRecordUpload) {
        chunkGeometryMetadataCache_.renderRecordBuffer->uploadToBuffer(commandBuffer);
        buffers.push_back(chunkGeometryMetadataCache_.renderRecordBuffer);
    }
    if (fullIndexAddressUpload) {
        chunkGeometryMetadataCache_.indexBufferAddressBuffer->uploadToBuffer(commandBuffer);
        buffers.push_back(chunkGeometryMetadataCache_.indexBufferAddressBuffer);
    }
    if (fullPbrAddressUpload) {
        chunkGeometryMetadataCache_.pbrBufferAddressBuffer->uploadToBuffer(commandBuffer);
        buffers.push_back(chunkGeometryMetadataCache_.pbrBufferAddressBuffer);
    }

    if (!fullRenderRecordUpload && uploadMetadataRanges(chunkGeometryMetadataCache_.renderRecordBuffer, commandBuffer,
                                                        chunkGeometryMetadataCache_.renderRecords,
                                                        chunkGeometryMetadataCache_.renderRecordDirtyRanges)) {
        buffers.push_back(chunkGeometryMetadataCache_.renderRecordBuffer);
    }
    if (!fullIndexAddressUpload && uploadMetadataRanges(chunkGeometryMetadataCache_.indexBufferAddressBuffer,
                                                        commandBuffer, chunkGeometryMetadataCache_.indexBufferAddresses,
                                                        chunkGeometryMetadataCache_.geometryDirtyRanges)) {
        buffers.push_back(chunkGeometryMetadataCache_.indexBufferAddressBuffer);
    }
    if (!fullPbrAddressUpload && uploadMetadataRanges(chunkGeometryMetadataCache_.pbrBufferAddressBuffer, commandBuffer,
                                                      chunkGeometryMetadataCache_.pbrBufferAddresses,
                                                      chunkGeometryMetadataCache_.geometryDirtyRanges)) {
        buffers.push_back(chunkGeometryMetadataCache_.pbrBufferAddressBuffer);
    }

    chunkGeometryMetadataCache_.renderRecordDirtyRanges.clear();
    chunkGeometryMetadataCache_.geometryDirtyRanges.clear();
    chunkGeometryMetadataCache_.hasPendingGpuUpload = false;

    if (buffers.empty()) { return; }

    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    std::vector<vk::CommandBuffer::BufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    for (const auto &buffer : buffers) {
        barriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
    }
    commandBuffer->barriersBufferImage(barriers, {});
}

void Chunks::updateEntityLightData(const std::vector<ChunkEntityLightRange> &ranges,
                                   uint32_t globalLightCount,
                                   VkDeviceAddress globalLightBufferAddress) {
    MCVR_PROFILE_SCOPE("native.Chunks.updateEntityLightData");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (!Renderer::options.collectChunkEmission || chunkPackedData_.empty()) { return; }

    clearEntityLightDataLocked();
    chunkPackedData_[0].globalEntityLightCount = globalLightCount;
    chunkPackedData_[0].globalEntityLightBufferAddress = globalLightBufferAddress;
    entityLightDataChunkIndices_.reserve(ranges.size());

    for (const ChunkEntityLightRange &range : ranges) {
        if (range.lightCount == 0 || range.lightBufferAddress == 0 || range.chunkIndex >= chunkPackedData_.size()) {
            continue;
        }

        ChunkPackedData &data = chunkPackedData_[range.chunkIndex];
        const bool rowEmpty = data.geometryCount == 0 && data.lightCount == 0 && data.lightBufferAddress == 0;
        const bool rowMatchesOrigin =
            data.x == range.chunkOrigin.x && data.y == range.chunkOrigin.y && data.z == range.chunkOrigin.z;
        if (!rowEmpty && !rowMatchesOrigin) { continue; }

        if (rowEmpty) {
            data.x = range.chunkOrigin.x;
            data.y = range.chunkOrigin.y;
            data.z = range.chunkOrigin.z;
        }
        data.entityLightCount = range.lightCount;
        data.entityLightBufferAddress = range.lightBufferAddress;
        entityLightDataChunkIndices_.push_back(range.chunkIndex);
    }

    chunkPackedDataDirty_ = true;
}

void Chunks::queueChunkPackedDataUpload() {
    MCVR_PROFILE_SCOPE("native.Chunks.queueChunkPackedDataUpload");
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    uploadChunkPackedDataForCurrentFrameLocked();
}

std::shared_ptr<ChunkBuildScheduler> Chunks::chunkBuildScheduler() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    return chunkBuildScheduler_;
}

std::shared_ptr<std::vector<std::shared_ptr<vk::BLASBuilder>>> Chunks::importantBLASBuilders() {
    return importantBLASBuilders_;
}

std::shared_ptr<vk::DeviceLocalBuffer> Chunks::chunkPackedData() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (context->frameIndex >= chunkPackedDataBuffers_.size()) { return nullptr; }
    return chunkPackedDataBuffers_[context->frameIndex];
}

glm::ivec4 Chunks::chunkGridInfo() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    return glm::ivec4(sizeX_, sizeY_, sizeZ_, bottomSectionCoord_);
}

void Chunks::setChunkStorageSectionPos(glm::ivec3 sectionPos) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    chunkStorageSectionPos_ = sectionPos;
}

glm::ivec4 Chunks::chunkStorageSectionPos() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    return glm::ivec4(chunkStorageSectionPos_, 0);
}
