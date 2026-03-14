
#include "core/render/buffers.hpp"

#include "common/shared.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <random>

std::ostream &buffersCout() {
    return std::cout << "[Buffers] ";
}

std::ostream &buffersCerr() {
    return std::cerr << "[Buffers] ";
}

Buffers::Buffers(std::shared_ptr<Framework> framework) {
    uint32_t size = framework->swapchain()->imageCount();

    validOverlayIndex_.resize(size);
    overlayIndexVertexBuffer_.resize(size);

    overlayDrawUniformBuffer_.resize(size);
    overlayPostUniformBuffer_.resize(size);

    worldUniformBuffer_.resize(size);
    lastWorldUniformBuffer_.resize(size);
    skyUniformBuffer_.resize(size);
    textureMappingBuffer_.resize(size);
    exposureDataBuffer_.resize(size);
    lightMapUniformBuffer_.resize(size);
}

void Buffers::resetFrame() {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto &gc = framework->gc();

    validOverlayIndex_[context->frameIndex].clear();

    overlayNextID_ = 0;

    gc.collect(overlayDrawUniformQueue_);
    overlayDrawUniformQueue_ = std::make_shared<std::vector<vk::Data::OverlayUBO>>();

    gc.collect(overlayPostUniformQueue_);
    overlayPostUniformQueue_ = std::make_shared<std::vector<vk::Data::OverlayPostUBO>>();

    gc.collect(importantIndexVertexBuffer_);
    importantIndexVertexBuffer_ = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>();
}

uint32_t Buffers::allocateBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    validOverlayIndex_[context->frameIndex].insert(std::make_pair(overlayNextID_, -1));
    auto it = overlayIndexVertexBuffer_[context->frameIndex].find(overlayNextID_);
    if (it == overlayIndexVertexBuffer_[context->frameIndex].end()) {
        overlayIndexVertexBuffer_[context->frameIndex].emplace(std::make_pair(overlayNextID_, nullptr));
    }
    return overlayNextID_++;
}

void Buffers::initializeBuffer(uint32_t id, uint32_t size, VkBufferUsageFlags usageFlags) {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();

    auto frameIndex = framework->safeAcquireCurrentContext()->frameIndex;
    auto device = framework->device();
    auto vma = framework->vma();

    auto bufferIter = overlayIndexVertexBuffer_[context->frameIndex].find(id);
    if (!validOverlayIndex_[context->frameIndex].contains(id) ||
        bufferIter == overlayIndexVertexBuffer_[context->frameIndex].end()) {
        buffersCerr() << "The given buffer id: " << id << " is not allocated for buffer" << std::endl;
        exit(EXIT_FAILURE);
    }

    validOverlayIndex_[context->frameIndex].at(id) = size;

    auto buffer = overlayIndexVertexBuffer_[context->frameIndex].contains(id) ?
                      overlayIndexVertexBuffer_[context->frameIndex].at(id) :
                      nullptr;
    uint32_t currentSize = buffer == nullptr ? baseBlockSize : buffer->size();
    while (currentSize < size) currentSize *= 2;
    if (buffer == nullptr || currentSize != buffer->size()) {
        framework->gc().collect(buffer);
        overlayIndexVertexBuffer_[context->frameIndex].at(id) =
            vk::DeviceLocalBuffer::create(vma, device, currentSize, usageFlags);
    }
}

void Buffers::buildIndexBuffer(uint32_t dstId, int type, int drawMode, int vertexCount, int expectedIndexCount) {
    auto buildQuadIndices = [this, dstId, vertexCount, expectedIndexCount]<typename V>() {
        int indexCount = vertexCount / 4 * 6;
        if (indexCount != expectedIndexCount) { throw std::runtime_error("index count not match!"); }

        std::vector<V> indices;
        for (int i = 0; i < vertexCount; i += 4) {
            indices.push_back(i + 0);
            indices.push_back(i + 1);
            indices.push_back(i + 2);
            indices.push_back(i + 2);
            indices.push_back(i + 3);
            indices.push_back(i + 0);
        }

        queueOverlayUpload(reinterpret_cast<uint8_t *>(indices.data()), dstId);
    };

    switch (drawMode) {
        case 7: {
            switch (type) {
                case 0: {
                    buildQuadIndices.template operator()<uint16_t>();
                    break;
                }
                case 1: {
                    buildQuadIndices.template operator()<uint32_t>();
                    break;
                }
            }
            break;
        }

        default: {
            std::cout << "Get draw mode=" << drawMode << std::endl;
            throw std::runtime_error("not implemented yet");
        }
    }
}

void Buffers::queueOverlayUpload(uint8_t *srcPointer, uint32_t dstId) {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    auto buffer = overlayIndexVertexBuffer_[context->frameIndex].at(dstId);
    if (validOverlayIndex_[context->frameIndex].contains(dstId) && buffer != nullptr) {
        auto size = validOverlayIndex_[context->frameIndex].at(dstId);
        if (size > 0) { buffer->uploadToStagingBuffer(srcPointer, size, 0); }
    }
}

void Buffers::queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                        std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer) {
    Renderer::instance().framework()->safeAcquireCurrentContext();
    Renderer::instance().framework()->safeAcquireCurrentContext();
    importantIndexVertexBuffer_->push_back(vertexBuffer);
    importantIndexVertexBuffer_->push_back(indexBuffer);
}

void Buffers::performQueuedUpload() {
    auto frameIndex = Renderer::instance().framework()->safeAcquireCurrentContext()->frameIndex;
    std::shared_ptr<vk::CommandBuffer> cmdBuffer =
        Renderer::instance().framework()->safeAcquireCurrentContext()->uploadCommandBuffer;

    auto physicalDevice = Renderer::instance().framework()->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> uploadPreBufferBarriers, uploadPostBufferBarriers;

    for (auto [bufferId, size] : validOverlayIndex_[frameIndex]) {
        auto buffer = overlayIndexVertexBuffer_[frameIndex].at(bufferId);
        uploadPreBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
        uploadPostBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                            VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
    }

    for (auto buffer : *importantIndexVertexBuffer_) {
        uploadPreBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
        uploadPostBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
    }

    cmdBuffer->barriersBufferImage(uploadPreBufferBarriers, {});

    for (auto [bufferId, size] : validOverlayIndex_[frameIndex]) {
        auto buffer = overlayIndexVertexBuffer_[frameIndex].at(bufferId);
        if (size > 0) { buffer->uploadToBuffer(cmdBuffer, size, 0, 0); }
    }

    for (auto buffer : *importantIndexVertexBuffer_) { buffer->uploadToBuffer(cmdBuffer); }

    cmdBuffer->barriersBufferImage(uploadPostBufferBarriers, {});
}

void Buffers::appendOverlayDrawUniform(vk::Data::OverlayUBO &ubo) {
    auto frameIndex = Renderer::instance().framework()->safeAcquireCurrentContext()->frameIndex;

    glm::mat4 mapGLToVulkan(1.0f);
    mapGLToVulkan[1][1] = -1.0f;
    mapGLToVulkan[2][2] = 0.5f;
    mapGLToVulkan[3][2] = 0.5f;

    ubo.projectionMat = mapGLToVulkan * ubo.projectionMat;

    overlayDrawUniformQueue_->push_back(ubo);
}

void Buffers::appendOverlayPostUniform(vk::Data::OverlayPostUBO &ubo) {
    auto frameIndex = Renderer::instance().framework()->safeAcquireCurrentContext()->frameIndex;
    overlayPostUniformQueue_->push_back(ubo);
}

void Buffers::buildAndUploadOverlayUniformBuffer() {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();
    auto pipelineContext =
        Renderer::instance().framework()->pipeline()->acquirePipelineContext(framework->safeAcquireCurrentContext());

    if (overlayDrawUniformQueue_->size() > 0) {
        if (overlayDrawUniformBuffer_[context->frameIndex] == nullptr ||
            overlayDrawUniformBuffer_.size() < overlayDrawUniformQueue_->size() * sizeof(vk::Data::OverlayUBO)) {
            uint32_t currentSize = overlayDrawUniformBuffer_[context->frameIndex] == nullptr ?
                                       baseBlockSize :
                                       overlayDrawUniformBuffer_[context->frameIndex]->size();
            while (currentSize < overlayDrawUniformQueue_->size() * sizeof(vk::Data::OverlayUBO)) currentSize *= 2;
            framework->gc().collect(overlayDrawUniformBuffer_[context->frameIndex]);
            overlayDrawUniformBuffer_[context->frameIndex] = vk::HostVisibleBuffer::create(
                vma, device, currentSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }

        overlayDrawUniformBuffer_[context->frameIndex]->uploadToBuffer(
            overlayDrawUniformQueue_->data(), overlayDrawUniformQueue_->size() * sizeof(vk::Data::OverlayUBO), 0);
        pipelineContext->uiModuleContext->overlayDescriptorTable->bindBuffer(
            overlayDrawUniformBuffer_[context->frameIndex], 1, 0);
    }

    if (overlayPostUniformQueue_->size() > 0) {
        if (overlayPostUniformBuffer_[context->frameIndex] == nullptr ||
            overlayPostUniformBuffer_[context->frameIndex]->size() <
                overlayPostUniformQueue_->size() * sizeof(vk::Data::OverlayPostUBO)) {
            uint32_t currentSize = overlayPostUniformBuffer_[context->frameIndex] == nullptr ?
                                       baseBlockSize :
                                       overlayPostUniformBuffer_[context->frameIndex]->size();
            while (currentSize < overlayPostUniformQueue_->size() * sizeof(vk::Data::OverlayPostUBO)) currentSize *= 2;
            framework->gc().collect(overlayPostUniformBuffer_[context->frameIndex]);
            overlayPostUniformBuffer_[context->frameIndex] = vk::HostVisibleBuffer::create(
                vma, device, currentSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }

        overlayPostUniformBuffer_[context->frameIndex]->uploadToBuffer(
            overlayPostUniformQueue_->data(), overlayPostUniformQueue_->size() * sizeof(vk::Data::OverlayPostUBO), 0);
        pipelineContext->uiModuleContext->overlayDescriptorTable->bindBuffer(
            overlayPostUniformBuffer_[context->frameIndex], 1, 1);
    }
}

static size_t sequenceIndex = 0;

// halton low discrepancy sequence, from https://www.shadertoy.com/view/wdXSW8
glm::vec2 halton(int index) {
    const glm::vec2 coprimes = glm::vec2(2.0F, 3.0F);
    glm::vec2 s = glm::vec2(index, index);
    glm::vec4 a = glm::vec4(1, 1, 0, 0);
    while (s.x > 0. && s.y > 0.) {
        a.x = a.x / coprimes.x;
        a.y = a.y / coprimes.y;
        a.z += a.x * fmod(s.x, coprimes.x);
        a.w += a.y * fmod(s.y, coprimes.y);
        s.x = floorf(s.x / coprimes.x);
        s.y = floorf(s.y / coprimes.y);
    }
    return glm::vec2(a.z, a.w);
}

void Buffers::setAndUploadWorldUniformBuffer(vk::Data::WorldUBO &ubo) {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    static vk::Data::WorldUBO lastUBO = []() {
        vk::Data::WorldUBO init{};
        init.cameraViewMat = glm::mat4(1.0f);
        init.cameraEffectedViewMat = glm::mat4(1.0f);
        init.cameraProjMat = glm::mat4(1.0f);
        init.cameraViewMatInv = glm::mat4(1.0f);
        init.cameraEffectedViewMatInv = glm::mat4(1.0f);
        init.cameraProjMatInv = glm::mat4(1.0f);
        return init;
    }();

    glm::mat4 mapGLToVulkan(1.0f);
    mapGLToVulkan[1][1] = -1.0f;
    mapGLToVulkan[2][2] = 0.5f;
    mapGLToVulkan[3][2] = 0.5f;

    ubo.cameraProjMat = mapGLToVulkan * ubo.cameraProjMat;

    ubo.cameraViewMatInv = glm::inverse(ubo.cameraViewMat);
    ubo.cameraEffectedViewMatInv = glm::inverse(ubo.cameraEffectedViewMat);
    ubo.cameraProjMatInv = glm::inverse(ubo.cameraProjMat);

    {
        static std::random_device seed;
        static std::ranlux48 engine(seed());
        static std::uniform_int_distribution<> distrib;
        ubo.seed = distrib(engine);
    }

    ubo.cameraJitter = useJitter_ ? halton(sequenceIndex++) - glm::vec2(0.5) : glm::vec2(0.0);

    ubo.rayBounces = Renderer::options.rayBounces;

    auto world = Renderer::instance().world();
    ubo.cameraPos.x = world->getCameraPos().x;
    ubo.cameraPos.y = world->getCameraPos().y;
    ubo.cameraPos.z = world->getCameraPos().z;
    ubo.cameraPos.w = 0;

    // VR stereo fields
    auto &vr = Renderer::instance().vrSystem();
    if (vr.enabled && vr.eyeCount > 1) {
        // VR path: headView (physical HMD tracking) * worldOrientation (game world rotation).
        // worldOrientation is set from Java each frame and encodes all game-world
        // rotation (player body yaw, stick/mouse turning, elytra roll, etc.).
        // The MC cameraViewMat is NOT used here — VR has its own rotation source.
        if (vr.headPose.valid) {
            glm::mat4 headView = vr.headPose.viewMatrix();
            glm::mat4 worldRot = glm::mat4_cast(glm::inverse(vr.worldOrientation));
            ubo.cameraViewMat = headView * worldRot;
            ubo.cameraEffectedViewMat = headView * worldRot;
            ubo.cameraViewMatInv = glm::inverse(ubo.cameraViewMat);
            ubo.cameraEffectedViewMatInv = glm::inverse(ubo.cameraEffectedViewMat);

            // Apply physical HMD offset (relative to worldPosition) to cameraPos.
            // This lets small head movements (lean, duck) shift the camera naturally.
            glm::vec3 trackingOffset = vr.headPose.position - vr.worldPosition;
            // Rotate tracking-space offset into game world space via worldOrientation
            glm::vec3 worldOffset = vr.worldOrientation * trackingOffset;
            worldOffset *= vr.config.worldScale;
            ubo.cameraPos.x += worldOffset.x;
            ubo.cameraPos.y += worldOffset.y;
            ubo.cameraPos.z += worldOffset.z;
        }

        ubo.stereoEnabled = 1;
        ubo.ipd = vr.config.ipd;
        for (uint32_t eye = 0; eye < 2; eye++) {
            ubo.eyeViewOffsets[eye] = vr.eyes[eye].viewOffset();
            // Compute per-eye projection correction relative to the base camera projection.
            // eyeProjOffsets[eye] = eyeProjection * inverse(cameraProjMat)
            // This lets the shader transform from the base projection to the per-eye asymmetric one.
            glm::mat4 eyeProj = vr.eyes[eye].projectionMatrix(0.05f, 1000.0f);
            ubo.eyeProjOffsets[eye] = eyeProj * glm::inverse(ubo.cameraProjMat);
        }
    } else {
        ubo.stereoEnabled = 0;
        ubo.ipd = 0.0f;
        for (uint32_t eye = 0; eye < 2; eye++) {
            ubo.eyeViewOffsets[eye] = glm::mat4(1.0f);
            ubo.eyeProjOffsets[eye] = glm::mat4(1.0f);
        }
    }

    ubo.foveatedInnerRadius = foveatedInnerRadius_;
    ubo.foveatedOuterBlockSize = foveatedOuterBlockSize_;

    // Foveated centre: use eye gaze if valid, otherwise screen centre
    if (vr.enabled && vr.gazeValid) {
        ubo.foveatedCenter = {vr.gazePoint.x, vr.gazePoint.y};
    } else {
        ubo.foveatedCenter = {0.5f, 0.5f};
    }

    if (worldUniformBuffer_[context->frameIndex] == nullptr) {
        worldUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::WorldUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    if (lastWorldUniformBuffer_[context->frameIndex] == nullptr) {
        lastWorldUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::WorldUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    worldUniformBuffer_[context->frameIndex]->uploadToBuffer(&ubo);
    lastWorldUniformBuffer_[context->frameIndex]->uploadToBuffer(&lastUBO);

    lastUBO = ubo;
}

void Buffers::setAndUploadSkyUniformBuffer(vk::Data::SkyUBO &ubo) {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    ubo.Rg = 6360000.0;
    ubo.Rt = 6460000.0;
    ubo.Hr = 8000.0;
    ubo.Hm = 1200.0;
    ubo.mieG = 0.80;
    ubo.betaR = glm::vec3(5.802e-6, 13.558e-6, 33.100e-6);
    ubo.betaM = glm::vec3(21.000e-6, 21.000e-6, 21.000e-6);
    ubo.minViewCos = 0.02;
    ubo.sunRadiance = glm::vec3(16);
    ubo.moonRadiance = glm::vec3(0.4, 0.5, 1);

    if (skyUniformBuffer_[context->frameIndex] == nullptr) {
        skyUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::SkyUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    skyUniformBuffer_[context->frameIndex]->uploadToBuffer(&ubo);
}

void Buffers::setAndUploadTextureMappingBuffer(vk::Data::TextureMapping &mapping) {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    if (textureMappingBuffer_[context->frameIndex] == nullptr) {
        textureMappingBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::TextureMapping),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    textureMappingBuffer_[context->frameIndex]->uploadToBuffer(&mapping);
}

void Buffers::setAndUploadExposureDataBuffer(vk::Data::ExposureData &exposureData) {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    if (exposureDataBuffer_[context->frameIndex] == nullptr) {
        exposureDataBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::ExposureData),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    exposureDataBuffer_[context->frameIndex]->uploadToBuffer(&exposureData);
}

void Buffers::setAndUploadLightMapUniformBuffer(vk::Data::LightMapUBO &ubo) {
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    if (lightMapUniformBuffer_[context->frameIndex] == nullptr) {
        lightMapUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::LightMapUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    lightMapUniformBuffer_[context->frameIndex]->uploadToBuffer(&ubo);
}

int Buffers::getDrawID() {
    Renderer::instance().framework()->safeAcquireCurrentContext();

    return overlayDrawUniformQueue_->size() - 1;
}

int Buffers::getPostID() {
    Renderer::instance().framework()->safeAcquireCurrentContext();

    return overlayPostUniformQueue_->size() - 1;
}

std::shared_ptr<vk::DeviceLocalBuffer> Buffers::getBuffer(uint32_t id) {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    auto bufferIter = overlayIndexVertexBuffer_[context->frameIndex].find(id);
    if (!validOverlayIndex_[context->frameIndex].contains(id) ||
        bufferIter == overlayIndexVertexBuffer_[context->frameIndex].end()) {
        buffersCerr() << "The given buffer id: " << id << " is not allocated for buffer" << std::endl;
        exit(EXIT_FAILURE);
    }

    return bufferIter->second;
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::overlayDrawUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    return overlayDrawUniformBuffer_[context->frameIndex];
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::overlayPostUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    return overlayPostUniformBuffer_[context->frameIndex];
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::worldUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (worldUniformBuffer_[context->frameIndex]) {
        return worldUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::lastWorldUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (lastWorldUniformBuffer_[context->frameIndex]) {
        return lastWorldUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::skyUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (skyUniformBuffer_[context->frameIndex]) {
        return skyUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::textureMappingBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (textureMappingBuffer_[context->frameIndex]) {
        return textureMappingBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::exposureDataBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (exposureDataBuffer_[context->frameIndex]) {
        return exposureDataBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::lightMapUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (lightMapUniformBuffer_[context->frameIndex]) {
        return lightMapUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

void Buffers::setUseJitter(bool useJitter) {
    useJitter_ = useJitter;
}

void Buffers::setFoveatedInnerRadius(float radius) {
    foveatedInnerRadius_ = radius;
}

void Buffers::setFoveatedOuterBlockSize(uint32_t blockSize) {
    foveatedOuterBlockSize_ = blockSize;
}