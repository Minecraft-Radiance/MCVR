#define STB_IMAGE_IMPLEMENTATION
#include "core/vulkan/image.hpp"

#include "core/vulkan/buffer.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/vma.hpp"

#include <cstring>
#include <iostream>
#include <sstream>

std::ostream &imageCout() {
    return std::cout << "[Image] ";
}

std::ostream &imageCerr() {
    return std::cerr << "[Image] ";
}

vk::SwapchainImage::SwapchainImage(
    std::shared_ptr<Device> device, VkImage image, uint32_t width, uint32_t height, VkFormat format)
    : device_(device), image_(image), width_(width), height_(height), format_(format) {
    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image_;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange = wholeColorSubresourceRange;

    if (vkCreateImageView(device_->vkDevice(), &createInfo, nullptr, &imageViews_[0]) != VK_SUCCESS) {
        imageCerr() << "failed to create image view for image" << std::endl;
        exit(EXIT_FAILURE);
    }
}

vk::SwapchainImage::~SwapchainImage() {
    for (int i = 0; i < imageViews_.size(); i++) { vkDestroyImageView(device_->vkDevice(), imageViews_[0], nullptr); }
}

uint32_t vk::SwapchainImage::width() {
    return width_;
}

uint32_t vk::SwapchainImage::height() {
    return height_;
}

uint32_t vk::SwapchainImage::layer() {
    return 1;
}

VkFormat &vk::SwapchainImage::vkFormat() {
    return format_;
}

VkImage &vk::SwapchainImage::vkImage() {
    return image_;
}

VkImageView &vk::SwapchainImage::vkImageView(int index) {
    return imageViews_[index];
}

VkImageLayout &vk::SwapchainImage::imageLayout() {
    return imageLayout_;
}

size_t vk::formatToByte(VkFormat format) {
    switch (format) {
        // 1 byte
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT: return 1;

        // 2 bytes
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SFLOAT: return 2;

        // 4 bytes
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_UINT:
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT: return 4;

        // 8 bytes
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT: return 8;

        // 12 bytes
        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32_SFLOAT: return 12;

        // 16 bytes
        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;

        default: {
            throw std::runtime_error("Format not allowed: " + std::to_string(format));
        }
    }
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage)
    : vk::DeviceLocalImage(device, vma, true, width, height, layer, format, usage) {}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage)
    : vk::DeviceLocalImage(
          device, vma, persistStaging, width, height, layer, format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {}


vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       VmaAllocationCreateFlags allocationFlags,
                                       VmaMemoryUsage vmaUsage,
                                       VkImageCreateFlags imageCreateFlags)
    : DeviceLocalImage(device,
                       vma,
                       persistStaging,
                       1,
                       width,
                       height,
                       layer,
                       format,
                       usage,
                       allocationFlags,
                       vmaUsage,
                       imageCreateFlags) {}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t mipLevels,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       VmaAllocationCreateFlags allocationFlags,
                                       VmaMemoryUsage vmaUsage,
                                       VkImageCreateFlags imageCreateFlags)
    : device_(device),
      vma_(vma),
      width_(width),
      height_(height),
      layer_(layer),
      format_(format),
      persistStaging_(persistStaging),
      usage_(usage),
      allocationFlags_(allocationFlags),
      vmaUsage_(vmaUsage) {
#ifdef DEBUG
    imageCout() << "Creating image with width: " << width << " height: " << height << " layer: " << layer
                << " channel: " << vk::formatToByte(format) << " mip level: " << mipLevels
                << " staging: " << (persistStaging ? "enabled" : "disabled") << std::endl;
#endif

    if (persistStaging_) {
        // staging buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = width_ * height_ * layer_ * vk::formatToByte(format);
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
                            &stagingAllocationInfo_) != VK_SUCCESS) {
            imageCerr() << "failed to create staging buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    // image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = imageCreateFlags;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = layer_;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | usage_;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.flags = allocationFlags_;
    allocationInfo.usage = vmaUsage;
    if (vmaCreateImage(vma_->allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, &allocationInfo_) !=
        VK_SUCCESS) {
        imageCerr() << "failed to create image" << std::endl;
        exit(EXIT_FAILURE);
    }

    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image_;
    createInfo.viewType = layer_ == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    createInfo.format = format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    if (usage_ == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        createInfo.subresourceRange = wholeDepthSubresourceRange;
    } else {
        createInfo.subresourceRange = wholeColorSubresourceRange;
    }

    if (vkCreateImageView(device_->vkDevice(), &createInfo, nullptr, &imageViews_[0]) != VK_SUCCESS) {
        imageCerr() << "failed to create image view for image" << std::endl;
        exit(EXIT_FAILURE);
    }
}

vk::DeviceLocalImage::~DeviceLocalImage() {
    for (int i = 0; i < imageViews_.size(); i++) { vkDestroyImageView(device_->vkDevice(), imageViews_[i], nullptr); }
    vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
    vmaDestroyImage(vma_->allocator(), image_, allocation_);

#ifdef DEBUG
    imageCout() << "device local image deconstructed" << std::endl;
#endif
}

void vk::DeviceLocalImage::uploadToStagingBuffer(void *src) {
    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            imageCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // staging buffer
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = width_ * height_ * layer_ * vk::formatToByte(format_);
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
                            &stagingAllocationInfo_) != VK_SUCCESS) {
            imageCerr() << "failed to create staging buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    size_t size = width_ * height_ * layer_ * vk::formatToByte(format_);
#ifdef DEBUG
    imageCout() << "Flushed " << size << " bytes into staging buffer" << std::endl;
#endif
    std::memcpy(mappedPtr_, src, size);
    vmaFlushAllocation(vma_->allocator(), stagingAllocation_, 0, size);

    if (!persistStaging_) {
        vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        mappedPtr_ = nullptr;
    }
}

void vk::DeviceLocalImage::uploadToImage(VkCommandBuffer cmdBuffer) {
    VkBufferImageCopy region = {};
    region.imageSubresource = {usage_ == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT ?
                                   static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) :
                                   static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT),
                               0, 0, layer_};
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer_, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void vk::DeviceLocalImage::uploadToImage(std::shared_ptr<CommandBuffer> cmdBuffer) {
    uploadToImage(cmdBuffer->vkCommandBuffer());
}

void vk::DeviceLocalImage::uploadToImage(VkCommandBuffer cmdBuffer,
                                         std::shared_ptr<Buffer> buffer,
                                         std::vector<VkBufferImageCopy> &regions) {
    vkCmdCopyBufferToImage(cmdBuffer, buffer->vkBuffer(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions.size(),
                           regions.data());
}

void vk::DeviceLocalImage::uploadToImage(std::shared_ptr<CommandBuffer> cmdBuffer,
                                         std::shared_ptr<Buffer> buffer,
                                         std::vector<VkBufferImageCopy> &regions) {
    uploadToImage(cmdBuffer->vkCommandBuffer(), buffer, regions);
}

uint32_t vk::DeviceLocalImage::width() {
    return width_;
}

uint32_t vk::DeviceLocalImage::height() {
    return height_;
}

uint32_t vk::DeviceLocalImage::layer() {
    return layer_;
}

VkFormat &vk::DeviceLocalImage::vkFormat() {
    return format_;
}

VkBuffer &vk::DeviceLocalImage::vkStagingBuffer() {
    return stagingBuffer_;
}

VkImage &vk::DeviceLocalImage::vkImage() {
    return image_;
}

VkImageView &vk::DeviceLocalImage::vkImageView(int index) {
    return imageViews_[index];
}

VkImageLayout &vk::DeviceLocalImage::imageLayout() {
    return imageLayout_;
}

void *vk::DeviceLocalImage::mappedPtr() {
    return mappedPtr_;
}

void vk::DeviceLocalImage::addImageView(VkImageViewCreateInfo info) {
    VkImageView vkImageView{};
    if (vkCreateImageView(device_->vkDevice(), &info, nullptr, &vkImageView) != VK_SUCCESS) {
        imageCerr() << "failed to create image view for image" << std::endl;
        exit(EXIT_FAILURE);
    }
    imageViews_.push_back(vkImageView);
}

void vk::DeviceLocalImage::createPerLayerViews() {
    if (layer_ <= 1) return;
    for (uint32_t i = 0; i < layer_; i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image_;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format_;
        viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1};
        addImageView(viewInfo);
    }
}

VkImageView vk::DeviceLocalImage::perLayerView(uint32_t layerIndex) {
    // Per-layer views start at index 1 (index 0 is the whole-image view)
    uint32_t viewIndex = 1 + layerIndex;
    if (viewIndex >= imageViews_.size()) {
        imageCerr() << "perLayerView: layerIndex " << layerIndex << " out of range" << std::endl;
        exit(EXIT_FAILURE);
    }
    return imageViews_[viewIndex];
}

vk::Sampler::Sampler(std::shared_ptr<Device> device)
    : Sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT) {}

vk::Sampler::Sampler(std::shared_ptr<Device> device,
                     VkFilter samplingMode,
                     VkSamplerMipmapMode mipmapMode,
                     VkSamplerAddressMode addressMode)
    : device_(device), samplingMode_(samplingMode), mipmapMode_(mipmapMode), addressMode_(addressMode) {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = samplingMode;                       // 放大时的过滤方式
    samplerInfo.minFilter = samplingMode;                       // 缩小时的过滤方式
    samplerInfo.addressModeU = addressMode;                     // U方向寻址
    samplerInfo.addressModeV = addressMode;                     // V方向寻址
    samplerInfo.addressModeW = addressMode;                     // W方向寻址
    samplerInfo.anisotropyEnable = VK_FALSE;                    // 先不启用各向异性过滤
    samplerInfo.maxAnisotropy = 16.0f;                          // 最大各向异性采样数
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK; // 边界色
    samplerInfo.unnormalizedCoordinates = VK_FALSE;             // 使用标准化坐标 [0,1]
    samplerInfo.compareEnable = VK_FALSE;                       // 禁用深度比较
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = mipmapMode;    // mipmap插值
    samplerInfo.mipLodBias = 0.0f;          // mipmap偏移
    samplerInfo.minLod = 0.0f;              // 最小mip层级
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE; // 最大mip层级（无限制）

    if (vkCreateSampler(device->vkDevice(), &samplerInfo, nullptr, &samper_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler!");
    }
}

vk::Sampler::~Sampler() {
    vkDestroySampler(device_->vkDevice(), samper_, nullptr);
}

VkSampler vk::Sampler::vkSamper() {
    return samper_;
}

VkFilter vk::Sampler::vkSamplingMode() {
    return samplingMode_;
}

VkSamplerMipmapMode vk::Sampler::vkMipmapMode() {
    return mipmapMode_;
}

VkSamplerAddressMode vk::Sampler::vkAddressMode() {
    return addressMode_;
}

std::ostream &imageLoaderCout() {
    return std::cout << "[ImageLoader] ";
}

std::ostream &imageLoaderCerr() {
    return std::cerr << "[ImageLoader] ";
}

// 0~1 float
float linearToSrgb(float linear) {
    if (linear <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }
}

// 0~255 uint8_t
uint8_t linearToSrgb(uint8_t linear) {
    if (linear / 255.0f <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return (1.055f * powf(linear / 255.0f, 1.0f / 2.4f) - 0.055f) * 255.0f;
    }
}

vk::ImageLoader::ImageLoader(std::vector<std::string> imagePaths, uint32_t forceChannel)
    : imagePaths_(imagePaths), channel_(forceChannel), layer_(imagePaths.size()), data_() {
    if (imagePaths.size() == 0) { imageLoaderCerr() << "Cannot load 0 image" << std::endl; }

    for (int i = 0; i < imagePaths_.size(); i++) {
        stbi_uc *imageData;
        int channel;
        if (i == 0) {
            imageData = stbi_load(imagePaths_[i].c_str(), &width_, &height_, &channel, 0);
            imageLoaderCout() << "Loaded image from " << imagePaths_[i] << " with width: " << width_
                              << " height: " << height_ << " channel: " << channel << std::endl;
        } else {
            int currentWidth, currentHeight;
            imageData = stbi_load(imagePaths_[i].c_str(), &currentWidth, &currentHeight, &channel, 0);
            if (currentWidth != width_ || currentHeight != height_) {
                imageLoaderCerr() << "images are not with the same shape" << std::endl;
                imageLoaderCerr() << "current: [width=" << currentWidth << ", height=" << currentHeight << "]"
                                  << std::endl;
                imageLoaderCerr() << "existing: [width=" << width_ << ", height=" << height_ << "]" << std::endl;
                exit(EXIT_FAILURE);
            }
            imageLoaderCout() << "Loaded image from " << imagePaths_[i] << " with width: " << currentWidth
                              << " height: " << currentHeight << " channel: " << channel << std::endl;
        }

        if (forceChannel < channel) {
            imageLoaderCerr() << "Cannot compress image" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int w = 0; w < width_; w++) {
            for (int h = 0; h < height_; h++) {
                for (int c = 0; c < std::min(channel, 3);
                     c++) { // only do linear to srgb transform for color, not alpha
                    data_.push_back(linearToSrgb(imageData[(w * height_ + h) * channel + c]));
                }

                if (channel == 3 && forceChannel == 4) {
                    data_.push_back(255);
                } else if (channel == 1 && forceChannel == 4) {
                    data_.push_back(linearToSrgb(imageData[(w * height_ + h) * channel + 0]));
                    data_.push_back(linearToSrgb(imageData[(w * height_ + h) * channel + 0]));
                    data_.push_back(255);
                } else if (channel == 4 && forceChannel == 4) {
                    data_.push_back(imageData[(w * height_ + h) * channel + 3]);
                } else {
                    imageLoaderCerr() << "Force channel of " << forceChannel << " is not support for channel "
                                      << channel << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }

        stbi_image_free(imageData);
    }
}

vk::ImageLoader::~ImageLoader() {}

uint32_t vk::ImageLoader::width() {
    return width_;
}

uint32_t vk::ImageLoader::height() {
    return height_;
}

uint32_t vk::ImageLoader::channel() {
    return channel_;
}

uint32_t vk::ImageLoader::layer() {
    return layer_;
}

void *vk::ImageLoader::data() {
    return data_.data();
}
