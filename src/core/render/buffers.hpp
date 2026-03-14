
#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <map>
#include <set>
#include <vector>

class Framework;

class Buffers : public SharedObject<Buffers> {
  public:
    Buffers(std::shared_ptr<Framework> framework);

    void resetFrame();
    uint32_t allocateBuffer();
    void initializeBuffer(uint32_t id, uint32_t size, VkBufferUsageFlags usageFlags);
    void buildIndexBuffer(uint32_t dstId, int type, int drawMode, int vertexCount, int expectedIndexCount);
    void queueOverlayUpload(uint8_t *srcPointer, uint32_t dstId);
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                   std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer);
    void performQueuedUpload();

    void appendOverlayDrawUniform(vk::Data::OverlayUBO &ubo);
    void appendOverlayPostUniform(vk::Data::OverlayPostUBO &ubo);
    void buildAndUploadOverlayUniformBuffer();

    void setAndUploadWorldUniformBuffer(vk::Data::WorldUBO &ubo);
    void setAndUploadSkyUniformBuffer(vk::Data::SkyUBO &ubo);
    void setAndUploadTextureMappingBuffer(vk::Data::TextureMapping &mapping);
    void setAndUploadExposureDataBuffer(vk::Data::ExposureData &exposureData);
    void setAndUploadLightMapUniformBuffer(vk::Data::LightMapUBO &ubo);

    int getDrawID();
    int getPostID();

    std::shared_ptr<vk::DeviceLocalBuffer> getBuffer(uint32_t id);

    std::shared_ptr<vk::HostVisibleBuffer> overlayDrawUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> overlayPostUniformBuffer();

    std::shared_ptr<vk::HostVisibleBuffer> worldUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> lastWorldUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> skyUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> textureMappingBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> exposureDataBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> lightMapUniformBuffer();

    void setUseJitter(bool useJitter);
    void setFoveatedInnerRadius(float radius);
    void setFoveatedOuterBlockSize(uint32_t blockSize);

  private:
    static constexpr uint32_t baseBlockSize = 16 * 1024;

    std::vector<std::map<uint32_t, int32_t>> validOverlayIndex_;
    std::vector<std::map<uint32_t, std::shared_ptr<vk::DeviceLocalBuffer>>> overlayIndexVertexBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> overlayDrawUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> overlayPostUniformBuffer_;
    uint32_t overlayNextID_;

    std::shared_ptr<std::vector<vk::Data::OverlayUBO>> overlayDrawUniformQueue_;
    std::shared_ptr<std::vector<vk::Data::OverlayPostUBO>> overlayPostUniformQueue_;

    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> worldUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> lastWorldUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> skyUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> textureMappingBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> exposureDataBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> lightMapUniformBuffer_;

    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> importantIndexVertexBuffer_;

    bool useJitter_ = true;
    float foveatedInnerRadius_ = 1.0f;
    uint32_t foveatedOuterBlockSize_ = 2;
};
