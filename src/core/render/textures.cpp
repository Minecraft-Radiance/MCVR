#include "core/render/textures.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/emission.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/texture_upload_order.hpp"
#include "core/vulkan/result.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

std::ostream &texturesCerr() {
    return std::cerr << "[Textures] ";
}

namespace {
bool isEightBitColorFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SRGB:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SRGB:
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB: return true;
        default: return false;
    }
}

void convertPixel(const uint8_t *source, size_t sourceBytesPerPixel, uint8_t *destination,
                  size_t destinationBytesPerPixel) {
    if (sourceBytesPerPixel == 0 || sourceBytesPerPixel > 4 || destinationBytesPerPixel == 0 ||
        destinationBytesPerPixel > 4) {
        throw std::runtime_error("Unsupported texture upload pixel layout");
    }

    if (sourceBytesPerPixel <= 2 && destinationBytesPerPixel >= 3) {
        destination[0] = source[0];
        destination[1] = source[0];
        destination[2] = source[0];
        if (destinationBytesPerPixel == 4) { destination[3] = sourceBytesPerPixel == 2 ? source[1] : 255; }
        return;
    }

    size_t sharedBytes = std::min(sourceBytesPerPixel, destinationBytesPerPixel);
    std::memcpy(destination, source, sharedBytes);
    for (size_t byte = sharedBytes; byte < destinationBytesPerPixel; ++byte) {
        destination[byte] = byte == 3 || (byte == 1 && sourceBytesPerPixel == 1) ? 255 : 0;
    }
}

size_t checkedMultiply(size_t left, size_t right, const char *message) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) { throw std::runtime_error(message); }
    return left * right;
}

size_t checkedAdd(size_t left, size_t right, const char *message) {
    if (right > std::numeric_limits<size_t>::max() - left) { throw std::runtime_error(message); }
    return left + right;
}
}

Textures::Textures(std::shared_ptr<Framework> framework) : framework_(std::move(framework)) {}

std::shared_ptr<vk::Sampler> Textures::getOrCreateSampler(std::shared_ptr<vk::Device> device,
                                                          VkFilter samplingMode,
                                                          VkSamplerMipmapMode mipmapMode,
                                                          VkSamplerAddressMode addressMode) {
    auto key = std::make_tuple(samplingMode, mipmapMode, addressMode);
    auto cachedSampler = samplerCache_.find(key);
    if (cachedSampler != samplerCache_.end()) { return cachedSampler->second; }

    auto sampler = vk::Sampler::create(device, samplingMode, mipmapMode, addressMode);
    samplerCache_.emplace(std::move(key), sampler);
    return sampler;
}

Textures::~Textures() {
    try {
        shutdown();
    } catch (const std::exception &error) {
        texturesCerr() << "Texture shutdown failed: " << error.what() << std::endl;
    } catch (...) {
        texturesCerr() << "Texture shutdown failed" << std::endl;
    }
}

void Textures::ensureRunning() const {
    if (framework_ == nullptr || shuttingDown_.load(std::memory_order_acquire)) {
        throw std::runtime_error("Texture renderer is closed");
    }
}

std::shared_ptr<Framework> Textures::frameworkOrThrow() const {
    ensureRunning();
    return framework_;
}

void Textures::reset() {
    MCVR_PROFILE_SCOPE("native.Textures.reset");
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    flushQueuedUploads(UploadFlushReason::TextureReinitialize);
    framework->waitDeviceIdle();
    activeUploadSubmissions_.clear();
    queuedUploadPages_.clear();
    queuedUploadBatches_.clear();
    queuedTextureClears_.clear();
    freeUploadCommandBuffers_.clear();
    freeUploadBuffers_.clear();
    freeUploadFences_.clear();
    freeUploadBufferBytes_ = 0;
    queuedUploadBytes_ = 0;
    textures_.clear();
    samplers.clear();
    {
        std::scoped_lock lck(emissionMtx_);
        if (emission_ != nullptr) { emission_->reset(); }
    }
    releasedTextureFallbackIds_.clear();
    nextID = 0;
}

void Textures::shutdown() {
    MCVR_PROFILE_SCOPE("native.Textures.shutdown");
    if (shuttingDown_.exchange(true, std::memory_order_acq_rel)) { return; }
    auto framework = framework_;
    if (framework == nullptr) return;

    std::scoped_lock lck(mtx_, framework->recreateMtx());
    try {
        framework->waitDeviceIdle();
    } catch (const std::exception &error) {
        texturesCerr() << "Failed to wait for texture uploads during shutdown: " << error.what() << std::endl;
    }
    activeUploadSubmissions_.clear();
    deferredUploadSubmissions_.clear();
    queuedUploadPages_.clear();
    queuedUploadBatches_.clear();
    queuedTextureClears_.clear();
    freeUploadCommandBuffers_.clear();
    freeUploadBuffers_.clear();
    freeUploadFences_.clear();
    freeUploadBufferBytes_ = 0;
    queuedUploadBytes_ = 0;
    textures_.clear();
    samplers.clear();
    samplerCache_.clear();
    releasedTextureFallbackIds_.clear();
    {
        std::scoped_lock emissionLock(emissionMtx_);
        if (emission_ != nullptr) { emission_->reset(); }
        emission_.reset();
    }
    nextID = 0;
}

void Textures::resetFrame() {
    MCVR_PROFILE_SCOPE("native.Textures.resetFrame");
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    recycleCompletedUploads();
    MCVR_PROFILE_METRIC("texture_upload.staging.cached_bytes", freeUploadBufferBytes_);
    MCVR_PROFILE_METRIC("texture_upload.staging.cached_pages", freeUploadBuffers_.size());
    MCVR_PROFILE_METRIC("texture_upload.staging.active_submissions", activeUploadSubmissions_.size());
}

uint32_t Textures::allocateTexture() {
    MCVR_PROFILE_SCOPE("native.Textures.allocateTexture");
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();

#ifdef MCVR_UI_BACKEND_OPENGL
    while (textures_.contains(nextID)) { nextID++; }
#endif
    if (nextID >= framework->device()->textureDescriptorCapacity()) {
        throw std::runtime_error("Texture descriptor capacity exhausted (" +
                                 std::to_string(framework->device()->textureDescriptorCapacity()) + ")");
    }
    textures_.emplace(std::make_pair(nextID, nullptr));
    samplers.emplace(std::make_pair(nextID, nullptr));
    return nextID++;
}

void Textures::initializeTexture(uint32_t id, uint32_t maxLevel, uint32_t width, uint32_t height, VkFormat format) {
    MCVR_PROFILE_SCOPE("native.Textures.initializeTexture");
    auto framework = frameworkOrThrow();
    auto device = framework->device();
    auto vma = framework->vma();

    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    if (maxLevel == 0 || width == 0 || height == 0) {
        throw std::runtime_error("Texture dimensions and mip level count must be positive");
    }

    submitDeferredUploadsLocked();
    bool hasQueuedUploadForTexture = queuedTextureClears_.contains(id);
    if (!hasQueuedUploadForTexture) {
        for (const auto &[batchKey, regions] : queuedUploadBatches_) {
            (void)regions;
            if (batchKey.first == id) {
                hasQueuedUploadForTexture = true;
                break;
            }
        }
    }
    if (hasQueuedUploadForTexture) { flushQueuedUploads(UploadFlushReason::TextureReinitialize); }

    auto textureIter = textures_.find(id);
    if (textureIter == textures_.end()) {
#ifdef MCVR_UI_BACKEND_OPENGL
        if (id >= device->textureDescriptorCapacity()) {
            throw std::runtime_error("OpenGL texture id exceeds supported native descriptor capacity");
        }
        textures_.emplace(std::make_pair(id, nullptr));
        samplers.emplace(std::make_pair(id, nullptr));
#else
        throw std::runtime_error("The given texture id is not allocated for texture");
#endif
    }

    framework->frameResourceRetainer().retain(textures_[id]);
    textures_[id] = vk::DeviceLocalImage::create(device, vma, false, maxLevel, width, height, 1, format,
                                                 VK_IMAGE_USAGE_SAMPLED_BIT, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0
#ifdef DEBUG
                                                 ,
                                                 "Texture " + std::to_string(id)
#endif
    );
    textures_[id]->setMemoryCategory(vk::ResourceMemoryCategory::Texture);

    auto samplerIter = samplers.find(id);
    if (samplerIter == samplers.end()) {
        throw std::runtime_error("The given texture id is not allocated for sampler");
    }
    framework->frameResourceRetainer().retain(samplers[id]);
    samplers[id] =
        getOrCreateSampler(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    framework->pipeline()->bindTexture(samplers[id], textures_[id], id);
    releasedTextureFallbackIds_.erase(id);
    for (const auto &[releasedId, fallbackId] : releasedTextureFallbackIds_) {
        if (fallbackId == id) {
            framework->pipeline()->bindTexture(samplers[id], textures_[id], releasedId);
        }
    }
}

void Textures::releaseTexture(uint32_t id) {
    MCVR_PROFILE_SCOPE("native.Textures.releaseTexture");
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();

    auto textureIter = textures_.find(id);
    auto samplerIter = samplers.find(id);
    if (textureIter == textures_.end() || samplerIter == samplers.end() || textureIter->second == nullptr) { return; }

    flushQueuedUploads(UploadFlushReason::TextureRelease);

    std::shared_ptr<vk::DeviceLocalImage> fallbackTexture;
    std::shared_ptr<vk::Sampler> fallbackSampler;
    uint32_t fallbackId = 0;
    for (const auto &[candidateId, candidateTexture] : textures_) {
        if (candidateId == id || candidateTexture == nullptr || releasedTextureFallbackIds_.contains(candidateId)) {
            continue;
        }
        auto candidateSampler = samplers.find(candidateId);
        if (candidateSampler == samplers.end() || candidateSampler->second == nullptr) { continue; }
        fallbackId = candidateId;
        fallbackTexture = candidateTexture;
        fallbackSampler = candidateSampler->second;
        break;
    }

    if (fallbackTexture == nullptr || fallbackSampler == nullptr) { return; }
    framework->pipeline()->bindTexture(fallbackSampler, fallbackTexture, id);
    releasedTextureFallbackIds_[id] = fallbackId;

    framework->frameResourceRetainer().retain(textureIter->second);
    framework->frameResourceRetainer().retain(samplerIter->second);
    textureIter->second = nullptr;
    samplerIter->second = nullptr;
    queuedTextureClears_.erase(id);
}

void Textures::queueTextureClear(uint32_t id, const VkClearColorValue &clearColor) {
    MCVR_PROFILE_SCOPE("native.Textures.queueTextureClear");
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    auto textureIter = textures_.find(id);
    if (textureIter == textures_.end() || textureIter->second == nullptr) {
        throw std::runtime_error("Cannot clear uninitialized texture");
    }
    flushQueuedUploads(UploadFlushReason::TextureReinitialize);
    queuedTextureClears_[id] = clearColor;
#if defined(MCVR_ENABLE_PROFILER)
    uploadFrameMetrics_.clearCount++;
#endif
}

void Textures::setSamplingMode(uint32_t id, VkFilter samplingMode, VkSamplerMipmapMode mipmapMode) {
    MCVR_PROFILE_SCOPE("native.Textures.setSamplingMode");
    auto framework = frameworkOrThrow();
    auto device = framework->device();

    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();

    auto samplerIter = samplers.find(id);
    auto textureIter = textures_.find(id);
    if (samplerIter == samplers.end() || samplerIter->second == nullptr || textureIter == textures_.end() ||
        textureIter->second == nullptr) {
        throw std::runtime_error("Cannot change sampling mode for an uninitialized texture");
    }
    auto sampler = samplerIter->second;
    if (sampler->vkSamplingMode() == samplingMode && sampler->vkMipmapMode() == mipmapMode) { return; }

    VkSamplerAddressMode addressMode = sampler->vkAddressMode();

    framework->frameResourceRetainer().retain(sampler);
    samplers[id] = getOrCreateSampler(device, samplingMode, mipmapMode, addressMode);
    framework->pipeline()->bindTexture(samplers[id], textures_[id], id);
}

void Textures::setAddressMode(uint32_t id, VkSamplerAddressMode addressMode) {
    MCVR_PROFILE_SCOPE("native.Textures.setAddressMode");
    auto framework = frameworkOrThrow();
    auto device = framework->device();

    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();

    auto samplerIter = samplers.find(id);
    auto textureIter = textures_.find(id);
    if (samplerIter == samplers.end() || samplerIter->second == nullptr || textureIter == textures_.end() ||
        textureIter->second == nullptr) {
        throw std::runtime_error("Cannot change address mode for an uninitialized texture");
    }
    auto sampler = samplerIter->second;
    if (sampler->vkAddressMode() == addressMode) { return; }

    VkFilter samplingMode = sampler->vkSamplingMode();
    VkSamplerMipmapMode mipmapMode = sampler->vkMipmapMode();

    framework->frameResourceRetainer().retain(sampler);
    samplers[id] = getOrCreateSampler(device, samplingMode, mipmapMode, addressMode);
    framework->pipeline()->bindTexture(samplers[id], textures_[id], id);
}

void Textures::queueUploads(const UploadRegionInfo *uploadRegions, uint32_t count) {
    MCVR_PROFILE_SCOPE("native.Textures.queueUploads");
    if (uploadRegions == nullptr || count == 0) { return; }

    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
#if defined(MCVR_ENABLE_PROFILER)
    uploadFrameMetrics_.batchCount++;
    if (count == 1) {
        uploadFrameMetrics_.singleRegionBatchCount++;
    } else {
        uploadFrameMetrics_.multiRegionBatchCount++;
    }
    uploadFrameMetrics_.maxRegionsPerBatch = std::max<uint64_t>(uploadFrameMetrics_.maxRegionsPerBatch, count);
#endif

    for (uint32_t i = 0; i < count; i++) {
        const UploadRegionInfo &uploadRegion = uploadRegions[i];
        constexpr uint32_t maxSignedJavaInt = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
        if (uploadRegion.sourceSizeBytes == 0 || uploadRegion.sourceSizeBytes > maxSignedJavaInt ||
            uploadRegion.sourceRowPixels == 0 || uploadRegion.sourceRowPixels > maxSignedJavaInt ||
            uploadRegion.width == 0 || uploadRegion.width > maxSignedJavaInt || uploadRegion.height == 0 ||
            uploadRegion.height > maxSignedJavaInt || uploadRegion.level > maxSignedJavaInt ||
            uploadRegion.sourceBytesPerPixel == 0 || uploadRegion.sourceBytesPerPixel > 4) {
            throw std::runtime_error("Texture upload contains invalid dimensions or source layout");
        }
        if (uploadRegion.textureId >= framework->device()->textureDescriptorCapacity()) {
            throw std::runtime_error("Texture upload targets an invalid texture id");
        }
        auto dstTextureIter = textures_.find(uploadRegion.textureId);
        if (dstTextureIter == textures_.end() || dstTextureIter->second == nullptr) {
            throw std::runtime_error("The destination texture is not initialized");
        }
        if (uploadRegion.uploadKind > 3) {
            throw std::runtime_error("Unknown texture upload kind");
        }
        if (uploadRegion.destinationOffsetX < 0 || uploadRegion.destinationOffsetY < 0) {
            throw std::runtime_error("Texture upload contains a negative destination offset");
        }

        auto texture = dstTextureIter->second;
        if (uploadRegion.level >= 32 || uploadRegion.level >= texture->fullSubresourceRange().levelCount) {
            throw std::runtime_error("Texture upload targets an unavailable mip level");
        }
        uint32_t levelWidth = std::max(1u, texture->width() >> uploadRegion.level);
        uint32_t levelHeight = std::max(1u, texture->height() >> uploadRegion.level);
        if (static_cast<uint64_t>(uploadRegion.destinationOffsetX) + uploadRegion.width > levelWidth ||
            static_cast<uint64_t>(uploadRegion.destinationOffsetY) + uploadRegion.height > levelHeight) {
            std::ostringstream message;
            message << "Texture upload region exceeds destination mip level: region=" << i
                    << ", textureId=" << uploadRegion.textureId << ", uploadKind=" << uploadRegion.uploadKind
                    << ", mip=" << uploadRegion.level << ", destinationOffset=" << uploadRegion.destinationOffsetX
                    << 'x' << uploadRegion.destinationOffsetY << ", uploadSize=" << uploadRegion.width << 'x'
                    << uploadRegion.height << ", destinationMipSize=" << levelWidth << 'x' << levelHeight
                    << ", destinationTextureSize=" << texture->width() << 'x' << texture->height()
                    << ", destinationMipLevels=" << texture->fullSubresourceRange().levelCount
                    << ", sourceOffset=" << uploadRegion.sourceOffsetX << 'x' << uploadRegion.sourceOffsetY
                    << ", sourceRowPixels=" << uploadRegion.sourceRowPixels
                    << ", sourceSizeBytes=" << uploadRegion.sourceSizeBytes
                    << ", sourceBytesPerPixel=" << uploadRegion.sourceBytesPerPixel;
            throw std::runtime_error(message.str());
        }

        size_t bytesPerPixel = vk::formatToByte(dstTextureIter->second->vkFormat());
        if (uploadRegion.sourceBytesPerPixel != bytesPerPixel &&
            !isEightBitColorFormat(dstTextureIter->second->vkFormat())) {
            throw std::runtime_error("Texture upload format conversion is unsupported for the destination format");
        }
        auto [pageIndex, offset] = appendUploadData(uploadRegion, bytesPerPixel);

        VkBufferImageCopy region = {};
        region.bufferRowLength = 0;
        region.bufferOffset = offset;
        region.imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        region.imageSubresource.mipLevel = uploadRegion.level;
        region.imageExtent = {uploadRegion.width, uploadRegion.height, 1};
        region.imageOffset = {uploadRegion.destinationOffsetX, uploadRegion.destinationOffsetY, 0};

        queuedUploadBatches_[{uploadRegion.textureId, pageIndex}].emplace_back(region);

        size_t uploadBytes = checkedMultiply(
            checkedMultiply(uploadRegion.width, uploadRegion.height, "Texture upload size overflows"), bytesPerPixel,
            "Texture upload size overflows");
        queuedUploadBytes_ = checkedAdd(queuedUploadBytes_, uploadBytes, "Queued texture upload size overflows");
#if defined(MCVR_ENABLE_PROFILER)
        queuedUploadRegionCount_++;
        uploadFrameMetrics_.regionCounts[uploadRegion.uploadKind]++;
        uploadFrameMetrics_.bytes[uploadRegion.uploadKind] += uploadBytes;
        uploadFrameMetrics_.stagingHighWaterBytes =
            std::max<uint64_t>(uploadFrameMetrics_.stagingHighWaterBytes, queuedUploadBytes_);
#endif
        if (queuedUploadBytes_ >= UPLOAD_FLUSH_THRESHOLD) { flushQueuedUploads(UploadFlushReason::Threshold); }
    }
}

void Textures::performQueuedUpload(bool deferSubmission) {
    MCVR_PROFILE_SCOPE("native.Textures.performQueuedUpload");
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    recycleCompletedUploads();
    flushQueuedUploads(UploadFlushReason::EndFrame, deferSubmission);
#if defined(MCVR_ENABLE_PROFILER)
    emitAndResetUploadFrameMetrics();
#endif
}

void Textures::submitDeferredUploads() {
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    submitDeferredUploadsLocked();
}

void Textures::submitDeferredUploadsLocked() {
    if (deferredUploadSubmissions_.empty()) { return; }
    MCVR_PROFILE_SCOPE("native.Textures.submitDeferredUploads");
    auto device = frameworkOrThrow()->device();
    activeUploadSubmissions_.reserve(activeUploadSubmissions_.size() + deferredUploadSubmissions_.size());
    while (!deferredUploadSubmissions_.empty()) {
        auto &submission = deferredUploadSubmissions_.front();
        submission.commandBuffer->submitMainQueueIndividual(device, submission.fence);
        activeUploadSubmissions_.push_back(std::move(submission));
        deferredUploadSubmissions_.erase(deferredUploadSubmissions_.begin());
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Textures::acquireUploadBuffer(size_t minSizeBytes) {
    auto framework = frameworkOrThrow();
    auto vma = framework->vma();
    auto device = framework->device();

#if defined(MCVR_ENABLE_PROFILER)
    uploadFrameMetrics_.stagingPageAcquireCount++;
    uploadFrameMetrics_.stagingPageRequestedBytes += minSizeBytes;
#endif
    auto bestIter = freeUploadBuffers_.end();
    for (auto iter = freeUploadBuffers_.begin(); iter != freeUploadBuffers_.end(); ++iter) {
        if ((*iter)->size() < minSizeBytes) { continue; }
        if (bestIter == freeUploadBuffers_.end() || (*iter)->size() < (*bestIter)->size()) { bestIter = iter; }
    }
    if (bestIter != freeUploadBuffers_.end()) {
        auto buffer = *bestIter;
        freeUploadBufferBytes_ -= std::min(freeUploadBufferBytes_, buffer->size());
        freeUploadBuffers_.erase(bestIter);
#if defined(MCVR_ENABLE_PROFILER)
        uploadFrameMetrics_.stagingPageReuseCount++;
        uploadFrameMetrics_.stagingPageCapacityBytes += buffer->size();
#endif
        return buffer;
    }

    size_t roundedSize = checkedAdd(minSizeBytes, UPLOAD_PAGE_GRANULARITY - 1,
                                    "Texture upload staging size overflows");
    size_t capacity = std::max(UPLOAD_PAGE_SIZE, roundedSize & ~(UPLOAD_PAGE_GRANULARITY - 1));
    auto buffer = vk::HostVisibleBuffer::create(vma, device, capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    buffer->setMemoryCategory(vk::ResourceMemoryCategory::Texture);
#if defined(MCVR_ENABLE_PROFILER)
    uploadFrameMetrics_.stagingPageCapacityBytes += capacity;
#endif
    return buffer;
}

void Textures::releaseUploadBuffer(std::shared_ptr<vk::HostVisibleBuffer> buffer) {
    if (buffer == nullptr) { return; }
    size_t bufferSize = buffer->size();
    if (bufferSize <= UPLOAD_STAGING_CACHE_BUDGET &&
        freeUploadBufferBytes_ <= UPLOAD_STAGING_CACHE_BUDGET - bufferSize) {
        freeUploadBufferBytes_ += bufferSize;
        freeUploadBuffers_.emplace_back(std::move(buffer));
        return;
    }
#if defined(MCVR_ENABLE_PROFILER)
    uploadFrameMetrics_.stagingPageDiscardCount++;
    uploadFrameMetrics_.stagingPageDiscardBytes += bufferSize;
#endif
}

std::shared_ptr<vk::Fence> Textures::acquireUploadFence() {
    auto framework = frameworkOrThrow();
    auto device = framework->device();

    if (!freeUploadFences_.empty()) {
        auto fence = freeUploadFences_.back();
        freeUploadFences_.pop_back();
        vk::requireSuccess(vkResetFences(device->vkDevice(), 1, &fence->vkFence()), "vkResetFences(texture upload)");
        return fence;
    }

    return vk::Fence::create(device);
}

void Textures::recycleCompletedUploads() {
    MCVR_PROFILE_SCOPE("native.Textures.recycleCompletedUploads");
    auto framework = frameworkOrThrow();
    auto device = framework->device();
    auto &submissions = activeUploadSubmissions_;
    for (auto iter = submissions.begin(); iter != submissions.end();) {
        VkResult fenceStatus = vkGetFenceStatus(device->vkDevice(), iter->fence->vkFence());
        if (fenceStatus == VK_SUCCESS) {
            freeUploadCommandBuffers_.emplace_back(iter->commandBuffer);
            freeUploadFences_.emplace_back(iter->fence);
            for (auto &buffer : iter->buffers) { releaseUploadBuffer(std::move(buffer)); }
            iter = submissions.erase(iter);
        } else if (fenceStatus == VK_NOT_READY) {
            ++iter;
        } else {
            vk::requireSuccess(fenceStatus, "vkGetFenceStatus(texture upload)");
        }
    }
}

std::pair<size_t, size_t> Textures::appendUploadData(const UploadRegionInfo &uploadRegion, size_t bytesPerPixel) {
    if (uploadRegion.sourcePointer == nullptr || uploadRegion.sourceOffsetX < 0 || uploadRegion.sourceOffsetY < 0 ||
        uploadRegion.destinationOffsetX < 0 || uploadRegion.destinationOffsetY < 0) {
        throw std::runtime_error("Texture upload contains a null source or negative offset");
    }

    const size_t sourceBytesPerPixel = uploadRegion.sourceBytesPerPixel;
    if (uploadRegion.width == 0 || uploadRegion.height == 0) {
        throw std::runtime_error("Texture upload region must not be empty");
    }
    if (sourceBytesPerPixel == 0 || sourceBytesPerPixel > 4 || uploadRegion.sourceRowPixels == 0) {
        throw std::runtime_error("Texture upload contains invalid source layout");
    }
    if (static_cast<uint64_t>(uploadRegion.sourceOffsetX) + uploadRegion.width > uploadRegion.sourceRowPixels) {
        throw std::runtime_error("Texture upload region exceeds source row width");
    }
    const size_t sourceRowBytes = checkedMultiply(uploadRegion.sourceRowPixels, sourceBytesPerPixel,
                                                  "Texture upload source row size overflows");
    const size_t regionSourceRowBytes = checkedMultiply(uploadRegion.width, sourceBytesPerPixel,
                                                        "Texture upload source region size overflows");
    const size_t regionRowBytes =
        checkedMultiply(uploadRegion.width, bytesPerPixel, "Texture upload destination row size overflows");
    const size_t sizeBytes =
        checkedMultiply(regionRowBytes, uploadRegion.height, "Texture upload destination size overflows");
    const size_t sourceRowOffset = checkedMultiply(static_cast<size_t>(uploadRegion.sourceOffsetY), sourceRowBytes,
                                                   "Texture upload source offset overflows");
    const size_t sourceColumnOffset =
        checkedMultiply(static_cast<size_t>(uploadRegion.sourceOffsetX), sourceBytesPerPixel,
                        "Texture upload source offset overflows");
    const size_t sourceOffsetBytes =
        checkedAdd(sourceRowOffset, sourceColumnOffset, "Texture upload source offset overflows");
    const size_t finalRowOffset = checkedMultiply(static_cast<size_t>(uploadRegion.height) - 1, sourceRowBytes,
                                                  "Texture upload source range overflows");
    const size_t finalRowStart =
        checkedAdd(sourceOffsetBytes, finalRowOffset, "Texture upload source range overflows");
    const size_t lastSourceByte =
        checkedAdd(finalRowStart, regionSourceRowBytes, "Texture upload source range overflows");
    if (lastSourceByte > uploadRegion.sourceSizeBytes) {
        throw std::runtime_error("Texture upload region is out of source bounds");
    }

    size_t writeOffset = 0;
    if (!queuedUploadPages_.empty()) {
        auto &lastPage = queuedUploadPages_.back();
        size_t alignedOffset = checkedAdd(lastPage.writeOffset, UPLOAD_OFFSET_ALIGNMENT - 1,
                                          "Texture upload staging offset overflows") &
                               ~(UPLOAD_OFFSET_ALIGNMENT - 1);
        if (alignedOffset > lastPage.buffer->size() || sizeBytes > lastPage.buffer->size() - alignedOffset) {
            writeOffset = 0;
        } else {
            writeOffset = alignedOffset;
        }
    }
    if (queuedUploadPages_.empty() || (writeOffset == 0 && queuedUploadPages_.back().writeOffset != 0)) {
        queuedUploadPages_.push_back({
            .buffer = acquireUploadBuffer(sizeBytes),
            .writeOffset = 0,
        });
        writeOffset = 0;
    }

    size_t pageIndex = queuedUploadPages_.size() - 1;
    auto &page = queuedUploadPages_[pageIndex];
    auto *destination = static_cast<uint8_t *>(page.buffer->mappedPtr()) + writeOffset;
    auto *source = uploadRegion.sourcePointer + sourceOffsetBytes;
    if (sourceBytesPerPixel == bytesPerPixel && uploadRegion.sourceOffsetX == 0 &&
        uploadRegion.width == uploadRegion.sourceRowPixels) {
        std::memcpy(destination, source, sizeBytes);
    } else if (sourceBytesPerPixel == bytesPerPixel) {
        for (uint32_t row = 0; row < uploadRegion.height; ++row) {
            std::memcpy(destination + static_cast<size_t>(row) * regionRowBytes,
                        source + static_cast<size_t>(row) * sourceRowBytes, regionRowBytes);
        }
    } else {
        for (uint32_t row = 0; row < uploadRegion.height; ++row) {
            auto *destinationRow = destination + static_cast<size_t>(row) * regionRowBytes;
            auto *sourceRow = source + static_cast<size_t>(row) * sourceRowBytes;
            for (uint32_t column = 0; column < uploadRegion.width; ++column) {
                convertPixel(sourceRow + static_cast<size_t>(column) * sourceBytesPerPixel,
                             sourceBytesPerPixel,
                             destinationRow + static_cast<size_t>(column) * bytesPerPixel, bytesPerPixel);
            }
        }
    }
    page.writeOffset = checkedAdd(writeOffset, sizeBytes, "Texture upload staging offset overflows");
    return {pageIndex, writeOffset};
}

void Textures::flushQueuedUploads(UploadFlushReason reason, bool deferSubmission) {
    MCVR_PROFILE_SCOPE("native.Textures.flushQueuedUploads");
    if (!deferSubmission) { submitDeferredUploadsLocked(); }
    if (queuedUploadBatches_.empty() && queuedTextureClears_.empty()) {
        queuedUploadBytes_ = 0;
#if defined(MCVR_ENABLE_PROFILER)
        queuedUploadRegionCount_ = 0;
#endif
        return;
    }

    auto framework = frameworkOrThrow();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    recycleCompletedUploads();
    deferredUploadSubmissions_.reserve(deferredUploadSubmissions_.size() + 1);
    std::shared_ptr<vk::CommandBuffer> cmdBuffer;
    if (!freeUploadCommandBuffers_.empty()) {
        cmdBuffer = freeUploadCommandBuffers_.back();
        freeUploadCommandBuffers_.pop_back();
        cmdBuffer->reset();
    } else {
        cmdBuffer = vk::CommandBuffer::create(device, framework->mainCommandPool());
    }
    auto fence = acquireUploadFence();
    cmdBuffer->begin();

    auto mainQueueIndex = physicalDevice->mainQueueIndex();

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> uploadPreImageBarriers, uploadPostImageBarriers;
    std::map<uint32_t, std::shared_ptr<vk::DeviceLocalImage>> targetTextures;
    for (const auto &[textureId, clearColor] : queuedTextureClears_) {
        (void)clearColor;
        targetTextures.emplace(textureId, nullptr);
    }
    for (const auto &[batchKey, regions] : queuedUploadBatches_) {
        (void)regions;
        targetTextures.emplace(batchKey.first, nullptr);
    }
    uploadPreImageBarriers.reserve(targetTextures.size());
    uploadPostImageBarriers.reserve(targetTextures.size());

    struct UploadLayoutRollback {
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> &barriers;
        bool armed = true;
        ~UploadLayoutRollback() {
            if (armed) {
                for (const auto &barrier : barriers) { barrier.image->imageLayout() = barrier.oldLayout; }
            }
        }
    } layoutRollback{uploadPreImageBarriers};

    for (auto &[textureId, texture] : targetTextures) {
        auto textureIter = textures_.find(textureId);
        if (textureIter == textures_.end() || textureIter->second == nullptr) {
            throw std::runtime_error("The textureId is not registered yet");
        }
        texture = textureIter->second;
        VkImageLayout oldLayout = texture->imageLayout();
        uploadPreImageBarriers.push_back({
            .srcStageMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                                                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask =
                oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = oldLayout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = texture,
            .subresourceRange = texture->fullSubresourceRange(),
        });
        texture->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        uploadPostImageBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = texture,
            .subresourceRange = texture->fullSubresourceRange(),
        });
        texture->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    cmdBuffer->barriersBufferImage({}, uploadPreImageBarriers);

    for (const auto &[textureId, clearColor] : queuedTextureClears_) {
        auto texture = targetTextures.at(textureId);
        VkImageSubresourceRange range = texture->fullSubresourceRange();
        vkCmdClearColorImage(cmdBuffer->vkCommandBuffer(), texture->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearColor, 1, &range);
    }

    if (!queuedTextureClears_.empty() && !queuedUploadBatches_.empty()) {
        cmdBuffer->barriersMemory({{
            .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        }});
    }

#if defined(MCVR_ENABLE_PROFILER)
    uint64_t copyBatchCount = 0;
    uint64_t copyBarrierCount = 0;
    uint64_t overlapCheckCount = 0;
#endif
    {
        MCVR_PROFILE_SCOPE("native.Textures.recordUploadCopies");
        for (auto batch = queuedUploadBatches_.begin(); batch != queuedUploadBatches_.end();) {
            const uint32_t textureId = batch->first.first;
            auto texture = targetTextures.at(textureId);
            TextureUploadOverlapTracker overlapTracker(texture->width(), texture->height());
            do {
                const size_t pageIndex = batch->first.second;
                const auto &regions = batch->second;
                if (pageIndex >= queuedUploadPages_.size()) {
                    throw std::runtime_error("Texture upload staging page is unavailable");
                }
                auto copyRegions = [&](size_t first, size_t end) {
                    if (first == end) { return; }
                    vkCmdCopyBufferToImage(cmdBuffer->vkCommandBuffer(),
                                           queuedUploadPages_[pageIndex].buffer->vkBuffer(), texture->vkImage(),
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(end - first),
                                           regions.data() + first);
#if defined(MCVR_ENABLE_PROFILER)
                    ++copyBatchCount;
#endif
                };
                size_t first = 0;
                for (size_t index = 0; index < regions.size(); ++index) {
                    const auto &region = regions[index];
#if defined(MCVR_ENABLE_PROFILER)
                    ++overlapCheckCount;
#endif
                    if (!overlapTracker.append(region.imageSubresource.mipLevel,
                                                static_cast<uint32_t>(region.imageOffset.x),
                                                static_cast<uint32_t>(region.imageOffset.y),
                                                region.imageExtent.width, region.imageExtent.height)) {
                        continue;
                    }
                    copyRegions(first, index);
                    cmdBuffer->barriersMemory({{
                        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    }});
#if defined(MCVR_ENABLE_PROFILER)
                    ++copyBarrierCount;
#endif
                    first = index;
                }
                copyRegions(first, regions.size());
                ++batch;
            } while (batch != queuedUploadBatches_.end() && batch->first.first == textureId);
        }
    }
#if defined(MCVR_ENABLE_PROFILER)
    MCVR_PROFILE_METRIC("texture_upload.flush.copy_batches", copyBatchCount);
    MCVR_PROFILE_METRIC("texture_upload.flush.copy_barriers", copyBarrierCount);
    MCVR_PROFILE_METRIC("texture_upload.flush.overlap_checks", overlapCheckCount);
#endif

    cmdBuffer->barriersBufferImage({}, uploadPostImageBarriers);

    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> buffers;
    buffers.reserve(queuedUploadPages_.size());
    for (auto &page : queuedUploadPages_) {
        page.buffer->flush(page.writeOffset, 0);
        buffers.emplace_back(page.buffer);
    }

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> images;
    images.reserve(targetTextures.size());
    for (auto &[textureId, texture] : targetTextures) {
        (void)textureId;
        images.emplace_back(std::move(texture));
    }

    cmdBuffer->end();
    deferredUploadSubmissions_.push_back({
        .fence = fence,
        .commandBuffer = cmdBuffer,
        .buffers = std::move(buffers),
        .images = std::move(images),
    });
    layoutRollback.armed = false;

#if defined(MCVR_ENABLE_PROFILER)
    const auto flushedUploadBytes = queuedUploadBytes_;
    const auto flushedUploadRegionCount = queuedUploadRegionCount_;
    const auto flushedUploadPageCount = queuedUploadPages_.size();
    const auto flushedTextureClearCount = queuedTextureClears_.size();
#endif

    queuedUploadBatches_.clear();
    queuedUploadPages_.clear();
    queuedTextureClears_.clear();
    queuedUploadBytes_ = 0;
#if defined(MCVR_ENABLE_PROFILER)
    queuedUploadRegionCount_ = 0;
#endif
    if (!deferSubmission) { submitDeferredUploadsLocked(); }

#if defined(MCVR_ENABLE_PROFILER)
    MCVR_PROFILE_METRIC("texture_upload.flush.reason", static_cast<uint32_t>(reason));
    MCVR_PROFILE_METRIC("texture_upload.flush.bytes", flushedUploadBytes);
    MCVR_PROFILE_METRIC("texture_upload.flush.regions", flushedUploadRegionCount);
    MCVR_PROFILE_METRIC("texture_upload.flush.textures", targetTextures.size());
    MCVR_PROFILE_METRIC("texture_upload.flush.pages", flushedUploadPageCount);
    MCVR_PROFILE_METRIC("texture_upload.flush.clears", flushedTextureClearCount);

    uploadFrameMetrics_.flushCount++;
    if (reason == UploadFlushReason::Threshold) uploadFrameMetrics_.thresholdFlushCount++;
    if (reason == UploadFlushReason::EndFrame) uploadFrameMetrics_.endFrameFlushCount++;
    if (reason == UploadFlushReason::TextureRelease || reason == UploadFlushReason::TextureReinitialize) {
        uploadFrameMetrics_.releaseFlushCount++;
    }
#endif
}

#if defined(MCVR_ENABLE_PROFILER)
void Textures::emitAndResetUploadFrameMetrics() {
    const auto &metrics = uploadFrameMetrics_;
    MCVR_PROFILE_METRIC("texture_upload.frame.batch_count", metrics.batchCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.single_region_batch_count", metrics.singleRegionBatchCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.multi_region_batch_count", metrics.multiRegionBatchCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.max_regions_per_batch", metrics.maxRegionsPerBatch);
    MCVR_PROFILE_METRIC("texture_upload.frame.main_regions", metrics.regionCounts[0]);
    MCVR_PROFILE_METRIC("texture_upload.frame.specular_regions", metrics.regionCounts[1]);
    MCVR_PROFILE_METRIC("texture_upload.frame.normal_regions", metrics.regionCounts[2]);
    MCVR_PROFILE_METRIC("texture_upload.frame.flag_regions", metrics.regionCounts[3]);
    MCVR_PROFILE_METRIC("texture_upload.frame.main_bytes", metrics.bytes[0]);
    MCVR_PROFILE_METRIC("texture_upload.frame.specular_bytes", metrics.bytes[1]);
    MCVR_PROFILE_METRIC("texture_upload.frame.normal_bytes", metrics.bytes[2]);
    MCVR_PROFILE_METRIC("texture_upload.frame.flag_bytes", metrics.bytes[3]);
    MCVR_PROFILE_METRIC("texture_upload.frame.aux_regions",
                        metrics.regionCounts[1] + metrics.regionCounts[2] + metrics.regionCounts[3]);
    MCVR_PROFILE_METRIC("texture_upload.frame.aux_bytes", metrics.bytes[1] + metrics.bytes[2] + metrics.bytes[3]);
    MCVR_PROFILE_METRIC("texture_upload.frame.flush_count", metrics.flushCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.threshold_flush_count", metrics.thresholdFlushCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.end_frame_flush_count", metrics.endFrameFlushCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.release_flush_count", metrics.releaseFlushCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.clear_count", metrics.clearCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_page_acquires", metrics.stagingPageAcquireCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_page_reuses", metrics.stagingPageReuseCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_page_discards", metrics.stagingPageDiscardCount);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_page_discard_bytes", metrics.stagingPageDiscardBytes);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_requested_bytes", metrics.stagingPageRequestedBytes);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_capacity_bytes", metrics.stagingPageCapacityBytes);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_high_water_bytes", metrics.stagingHighWaterBytes);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_growth_count", 0);
    MCVR_PROFILE_METRIC("texture_upload.frame.staging_growth_copy_bytes", 0);
    MCVR_PROFILE_METRIC("texture_upload.staging.cached_bytes", freeUploadBufferBytes_);
    MCVR_PROFILE_METRIC("texture_upload.staging.cached_pages", freeUploadBuffers_.size());
    MCVR_PROFILE_METRIC("texture_upload.staging.active_submissions", activeUploadSubmissions_.size());
    uploadFrameMetrics_ = {};
}
#endif

std::shared_ptr<vk::DeviceLocalImage> Textures::texture(uint32_t id) {
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    auto iter = textures_.find(id);
    return iter != textures_.end() ? iter->second : nullptr;
}

std::shared_ptr<vk::Sampler> Textures::sampler(uint32_t id) {
    auto framework = frameworkOrThrow();
    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();
    auto iter = samplers.find(id);
    return iter != samplers.end() ? iter->second : nullptr;
}

std::shared_ptr<Emission> Textures::emission() {
    MCVR_PROFILE_SCOPE("native.Textures.emission");
    if (!Renderer::options.collectChunkEmission || shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }

    std::scoped_lock lck(emissionMtx_);
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }
    if (emission_ == nullptr) { emission_ = Emission::create(std::weak_ptr<Textures>(shared_from_this())); }
    return emission_;
}

void Textures::releaseEmission() {
    MCVR_PROFILE_SCOPE("native.Textures.releaseEmission");
    std::scoped_lock lck(emissionMtx_);
    if (emission_ != nullptr) {
        emission_->reset();
        emission_ = nullptr;
    }
}

void Textures::bindAllTextures() {
    MCVR_PROFILE_SCOPE("native.Textures.bindAllTextures");
    auto framework = frameworkOrThrow();

    std::scoped_lock lck(mtx_, framework->recreateMtx());
    ensureRunning();

    for (const auto &[id, texture] : textures_) {
        if (texture == nullptr) { continue; }
        auto samplerIter = samplers.find(id);
        if (samplerIter == samplers.end() || samplerIter->second == nullptr) {
            throw std::runtime_error("Initialized texture has no sampler");
        }
        framework->pipeline()->bindTexture(samplerIter->second, texture, id);
    }
}
