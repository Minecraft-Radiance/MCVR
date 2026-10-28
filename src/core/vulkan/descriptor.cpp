#include "core/vulkan/descriptor.hpp"

#include "core/vulkan/as.hpp"
#include "core/vulkan/buffer.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

std::ostream &descriptorTableCerr() {
    return std::cerr << "[DescriptorTable] ";
}

struct PushConstantStageRange {
    bool hasRange = false;
    uint32_t offset = 0;
    uint32_t end = 0;
};

static constexpr std::array<VkShaderStageFlagBits, 14> pushConstantShaderStages = {
    VK_SHADER_STAGE_VERTEX_BIT,
    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
    VK_SHADER_STAGE_GEOMETRY_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VK_SHADER_STAGE_TASK_BIT_EXT,
    VK_SHADER_STAGE_MESH_BIT_EXT,
    VK_SHADER_STAGE_RAYGEN_BIT_KHR,
    VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
    VK_SHADER_STAGE_MISS_BIT_KHR,
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
    VK_SHADER_STAGE_CALLABLE_BIT_KHR,
};

static std::vector<VkPushConstantRange> normalizePushConstantRanges(const std::vector<VkPushConstantRange> &ranges) {
    std::array<PushConstantStageRange, pushConstantShaderStages.size()> stageRanges{};
    VkShaderStageFlags knownStages = 0;
    for (VkShaderStageFlagBits stage : pushConstantShaderStages) { knownStages |= stage; }

    for (const VkPushConstantRange &range : ranges) {
        if (range.stageFlags == 0 || range.size == 0) {
            descriptorTableCerr() << "invalid push constant range: stageFlags and size must be non-zero" << std::endl;
            exit(EXIT_FAILURE);
        }

        const VkShaderStageFlags unknownStages = range.stageFlags & ~knownStages;
        if (unknownStages != 0) {
            descriptorTableCerr() << "invalid push constant range: unsupported shader stage flags 0x" << std::hex
                                  << unknownStages << std::dec << std::endl;
            exit(EXIT_FAILURE);
        }

        const uint64_t rangeEnd = static_cast<uint64_t>(range.offset) + static_cast<uint64_t>(range.size);
        if (rangeEnd > std::numeric_limits<uint32_t>::max()) {
            descriptorTableCerr() << "invalid push constant range: offset + size overflows uint32_t" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (size_t stageIndex = 0; stageIndex < pushConstantShaderStages.size(); stageIndex++) {
            if ((range.stageFlags & pushConstantShaderStages[stageIndex]) == 0) { continue; }

            PushConstantStageRange &stageRange = stageRanges[stageIndex];
            if (!stageRange.hasRange) {
                stageRange.hasRange = true;
                stageRange.offset = range.offset;
                stageRange.end = static_cast<uint32_t>(rangeEnd);
                continue;
            }

            stageRange.offset = std::min(stageRange.offset, range.offset);
            stageRange.end = std::max(stageRange.end, static_cast<uint32_t>(rangeEnd));
        }
    }

    std::vector<VkPushConstantRange> normalized;
    for (size_t stageIndex = 0; stageIndex < pushConstantShaderStages.size(); stageIndex++) {
        const PushConstantStageRange &stageRange = stageRanges[stageIndex];
        if (!stageRange.hasRange) { continue; }

        const VkShaderStageFlags stageFlags = pushConstantShaderStages[stageIndex];
        const uint32_t size = stageRange.end - stageRange.offset;
        auto existingRange = std::find_if(normalized.begin(), normalized.end(), [&](const VkPushConstantRange &range) {
            return range.offset == stageRange.offset && range.size == size;
        });

        if (existingRange != normalized.end()) {
            existingRange->stageFlags |= stageFlags;
        } else {
            normalized.push_back({
                .stageFlags = stageFlags,
                .offset = stageRange.offset,
                .size = size,
            });
        }
    }

    return normalized;
}

vk::DescriptorTable::DescriptorTable(std::shared_ptr<Device> device,
                                     VkDescriptorPool descriptorPool,
                                     std::vector<VkDescriptorSetLayout> tableLayout,
                                     std::vector<VkDescriptorSet> table,
                                     std::vector<std::vector<VkDescriptorType>> tableTypes,
                                     std::vector<bool> pushDescriptorSets,
                                     std::vector<VkPushConstantRange> pushConstantRanges)
    : device_(device),
      descriptorPool_(descriptorPool),
      tableLayout_(tableLayout),
      table_(table),
      tableTypes_(tableTypes),
      pushDescriptorSets_(pushDescriptorSets),
      pushConstantRanges_(pushConstantRanges) {
    VkPipelineLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = tableLayout_.size();
    layoutCreateInfo.pSetLayouts = tableLayout_.data();
    layoutCreateInfo.pushConstantRangeCount = pushConstantRanges_.size();
    layoutCreateInfo.pPushConstantRanges = pushConstantRanges_.data();

    if (vkCreatePipelineLayout(device->vkDevice(), &layoutCreateInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        descriptorTableCerr() << "failed to create pipeline layout" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void vk::DescriptorTable::trackBoundResource(
    uint32_t set, uint32_t binding, uint32_t arrayElement, uint32_t resourceComponent, std::shared_ptr<void> resource) {
    auto key = std::make_tuple(set, binding, arrayElement, resourceComponent);
    auto resourceIndex = boundResourceIndices_.find(key);
    if (resourceIndex == boundResourceIndices_.end()) {
        boundResourceIndices_.emplace(std::move(key), boundResources_.size());
        boundResources_.emplace_back(std::move(resource));
        return;
    }
    boundResources_[resourceIndex->second] = std::move(resource);
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindImage(
    std::shared_ptr<Image> image, VkImageLayout layout, uint32_t set, uint32_t binding, uint32_t viewIndex) {
    if (!image) { throw std::runtime_error("DescriptorTable::bindImage received a null image"); }
    VkDescriptorType descriptorType = checkedDescriptorType(set, binding, "bindImage");

    VkDescriptorImageInfo descriptorImageInfo{};
    descriptorImageInfo.imageView = image->vkImageView(viewIndex);
    descriptorImageInfo.imageLayout = layout;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = descriptorType;
    writeDescriptorSet.pImageInfo = &descriptorImageInfo;
    writeDescriptorSet.dstBinding = binding;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    trackBoundResource(set, binding, 0, 0, image);

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindSamplerImage(std::shared_ptr<Sampler> sampler,
                                                                           std::shared_ptr<Image> image,
                                                                           VkImageLayout layout,
                                                                           uint32_t set,
                                                                           uint32_t binding,
                                                                           uint32_t index,
                                                                           uint32_t viewIndex) {
    if (!sampler) { throw std::runtime_error("DescriptorTable::bindSamplerImage received a null sampler"); }
    if (!image) { throw std::runtime_error("DescriptorTable::bindSamplerImage received a null image"); }
    VkDescriptorType descriptorType = checkedDescriptorType(set, binding, "bindSamplerImage");

    VkDescriptorImageInfo descriptorImageInfo{};
    descriptorImageInfo.sampler = sampler->vkSamper();
    descriptorImageInfo.imageView = image->vkImageView(viewIndex);
    descriptorImageInfo.imageLayout = layout;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = descriptorType;
    writeDescriptorSet.pImageInfo = &descriptorImageInfo;
    writeDescriptorSet.dstBinding = binding;
    writeDescriptorSet.dstArrayElement = index;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    trackBoundResource(set, binding, index, 0, sampler);
    trackBoundResource(set, binding, index, 1, image);

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindImageForShader(std::shared_ptr<Image> image,
                                                                             uint32_t set,
                                                                             uint32_t binding,
                                                                             uint32_t viewIndex) {
    return bindImage(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, set, binding, viewIndex);
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindSamplerImageForShader(std::shared_ptr<Sampler> sampler,
                                                                                    std::shared_ptr<Image> image,
                                                                                    uint32_t set,
                                                                                    uint32_t binding,
                                                                                    uint32_t viewIndex) {
    return bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, set, binding, 0, viewIndex);
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindBuffer(std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding) {
    return bindBufferRange(buffer, set, binding, 0, 0, buffer->size());
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindBuffer(std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding, uint32_t index) {
    return bindBufferRange(buffer, set, binding, index, 0, buffer->size());
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindBufferRange(
    std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding, VkDeviceSize offset, VkDeviceSize range) {
    return bindBufferRange(buffer, set, binding, 0, offset, range);
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindBufferRange(std::shared_ptr<Buffer> buffer,
                                                                          uint32_t set,
                                                                          uint32_t binding,
                                                                          uint32_t index,
                                                                          VkDeviceSize offset,
                                                                          VkDeviceSize range) {
    if (!buffer) { throw std::runtime_error("DescriptorTable::bindBufferRange received a null buffer"); }
    VkDescriptorType descriptorType = checkedDescriptorType(set, binding, "bindBufferRange");

    VkDescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.buffer = buffer->vkBuffer();
    descriptorBufferInfo.offset = offset;
    descriptorBufferInfo.range = range;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = descriptorType;
    writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
    writeDescriptorSet.dstBinding = binding;
    writeDescriptorSet.dstArrayElement = index;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);
    trackBoundResource(set, binding, index, 0, buffer);
    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindBuffers(std::vector<std::shared_ptr<Buffer>> buffers, uint32_t set, uint32_t binding) {
    VkDescriptorType descriptorType = checkedDescriptorType(set, binding, "bindBuffers");

    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
    for (int i = 0; i < buffers.size(); i++) {
        if (!buffers[i]) { throw std::runtime_error("DescriptorTable::bindBuffers received a null buffer"); }
        VkDescriptorBufferInfo descriptorBufferInfo{};
        descriptorBufferInfo.buffer = buffers[i]->vkBuffer();
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = buffers[i]->size();
        descriptorBufferInfos.push_back(descriptorBufferInfo);
    }

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = buffers.size();
    writeDescriptorSet.descriptorType = descriptorType;
    writeDescriptorSet.pBufferInfo = descriptorBufferInfos.data();
    writeDescriptorSet.dstBinding = binding;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    for (uint32_t index = 0; index < buffers.size(); ++index) {
        trackBoundResource(set, binding, index, 0, buffers[index]);
    }

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindAS(std::shared_ptr<TLAS> tlas, uint32_t set, uint32_t binding) {
    if (!tlas) { throw std::runtime_error("DescriptorTable::bindAS received a null TLAS"); }
    VkDescriptorType descriptorType = checkedDescriptorType(set, binding, "bindAS");

    VkWriteDescriptorSetAccelerationStructureKHR asWriteInfo{};
    asWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWriteInfo.accelerationStructureCount = 1;
    asWriteInfo.pAccelerationStructures = &tlas->tlas();

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.pNext = &asWriteInfo;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = descriptorType;
    writeDescriptorSet.dstBinding = binding;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    trackBoundResource(set, binding, 0, 0, tlas);

    return shared_from_this();
}

uint32_t vk::DescriptorTable::setCount() {
    return table_.size();
}

uint32_t vk::DescriptorTable::dynamicDescriptorCount() const {
    uint32_t count = 0;
    for (const auto &setTypes : tableTypes_) {
        for (VkDescriptorType type : setTypes) {
            if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
                count++;
            }
        }
    }
    return count;
}

std::vector<VkDescriptorSet> &vk::DescriptorTable::descriptorSet() {
    return table_;
}

std::vector<VkDescriptorSetLayout> &vk::DescriptorTable::descriptorSetLayout() {
    return tableLayout_;
}

VkPipelineLayout &vk::DescriptorTable::vkPipelineLayout() {
    return pipelineLayout_;
}

const std::vector<std::shared_ptr<void>> &vk::DescriptorTable::boundResources() const {
    return boundResources_;
}

bool vk::DescriptorTable::isPushDescriptorSet(uint32_t set) const {
    return set < pushDescriptorSets_.size() && pushDescriptorSets_[set];
}

VkDescriptorSetLayout vk::DescriptorTable::pushDescriptorSetLayout(uint32_t set) const {
    return set < tableLayout_.size() ? tableLayout_[set] : VK_NULL_HANDLE;
}

VkDescriptorType vk::DescriptorTable::descriptorType(uint32_t set, uint32_t binding, const char *operation) const {
    if (set >= tableTypes_.size()) {
        std::ostringstream error;
        error << "DescriptorTable::" << operation << " uses invalid set " << set;
        throw std::runtime_error(error.str());
    }
    if (binding >= tableTypes_[set].size() || tableTypes_[set][binding] == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
        std::ostringstream error;
        error << "DescriptorTable::" << operation << " uses undefined binding " << binding << " in set " << set;
        throw std::runtime_error(error.str());
    }
    return tableTypes_[set][binding];
}

VkDescriptorType
vk::DescriptorTable::checkedDescriptorType(uint32_t set, uint32_t binding, const char *operation) const {
    VkDescriptorType type = descriptorType(set, binding, operation);
    if (set >= table_.size() || isPushDescriptorSet(set) || table_[set] == VK_NULL_HANDLE) {
        std::ostringstream error;
        error << "DescriptorTable::" << operation << " cannot update push/null descriptor set " << set;
        throw std::runtime_error(error.str());
    }
    return type;
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder::DescriptorLayoutSetBindingBuilder(
    vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder &parent, bool isPushDescriptor)
    : parent(parent), bindings(), isPushDescriptor(isPushDescriptor) {}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder::defineDescriptorLayoutSetBinding(
    VkDescriptorSetLayoutBinding binding) {
    bindings.push_back(binding);
    types.push_back(binding.descriptorType);
    return *this;
}

vk::DescriptorTable::~DescriptorTable() {
    for (VkDescriptorSetLayout layout : tableLayout_) {
        vkDestroyDescriptorSetLayout(device_->vkDevice(), layout, nullptr);
    }
    vkDestroyPipelineLayout(device_->vkDevice(), pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(device_->vkDevice(), descriptorPool_, nullptr);
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder::endDescriptorLayoutSetBinding() {
    return parent;
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::DescriptorLayoutSetBuilder(DescriptorTableBuilder &parent)
    : parent(parent), setBindingBuilders() {}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::beginDescriptorLayoutSetBinding() {
    return setBindingBuilders.emplace_back(*this, false);
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::beginStaticDescriptorLayoutSetBinding() {
    auto &builder = setBindingBuilders.emplace_back(*this, false);
    builder.isStaticDescriptor = true;
    return builder;
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::beginPushDescriptorLayoutSetBinding() {
    return setBindingBuilders.emplace_back(*this, true);
}

vk::DescriptorTableBuilder &vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::endDescriptorLayoutSet() {
    return parent;
}

vk::DescriptorTableBuilder::DescriptorTableBuilder() : setBuilders(*this) {}

vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder &vk::DescriptorTableBuilder::beginDescriptorLayoutSet() {
    return setBuilders;
}

vk::DescriptorTableBuilder &vk::DescriptorTableBuilder::definePushConstant(VkPushConstantRange pushConstantRange) {
    pushConstantRanges.push_back(pushConstantRange);
    return *this;
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTableBuilder::build(std::shared_ptr<Device> device) {
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<bool> pushDescriptorSets;
    std::map<VkDescriptorType, uint32_t> descriptorTypeCount;
    bool hasUpdateAfterBindSet = false;

    for (vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &setBindingBuilder :
         setBuilders.setBindingBuilders) {
        VkDescriptorSetLayout &descriptorSetLayout = descriptorSetLayouts.emplace_back();
        pushDescriptorSets.push_back(setBindingBuilder.isPushDescriptor);

        if (setBindingBuilder.isPushDescriptor) {
            VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo = {};
            descriptorLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            descriptorLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setBindingBuilder.bindings.size());
            descriptorLayoutCreateInfo.pBindings = setBindingBuilder.bindings.data();

            if (vkCreateDescriptorSetLayout(device->vkDevice(), &descriptorLayoutCreateInfo, nullptr,
                                            &descriptorSetLayout) != VK_SUCCESS) {
                descriptorTableCerr() << "failed to create push descriptor layout" << std::endl;
                exit(EXIT_FAILURE);
            }
            continue;
        }

        bool hasDynamicDescriptor = false;
        for (const VkDescriptorSetLayoutBinding &binding : setBindingBuilder.bindings) {
            if (binding.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
                hasDynamicDescriptor = true;
                break;
            }
        }

        std::vector<VkDescriptorBindingFlags> bindingFlags(static_cast<uint32_t>(setBindingBuilder.bindings.size()), 0);
        const bool hasUnsupportedAccelerationStructure =
            !device->hasAccelerationStructureUpdateAfterBind() &&
            std::any_of(setBindingBuilder.bindings.begin(), setBindingBuilder.bindings.end(),
                        [](const VkDescriptorSetLayoutBinding &binding) {
                            return binding.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                        });
        // Acceleration structures in an update-after-bind pool also have separate
        // descriptor limits (zero on MageVK), so make their entire set static.
        const bool updateAfterBind = !hasDynamicDescriptor && !setBindingBuilder.isStaticDescriptor &&
                                     !hasUnsupportedAccelerationStructure;
        if (updateAfterBind) {
            std::fill(bindingFlags.begin(), bindingFlags.end(),
                      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            hasUpdateAfterBindSet = true;
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsInfo.bindingCount = static_cast<uint32_t>(setBindingBuilder.bindings.size());
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();

        VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo = {};
        descriptorLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorLayoutCreateInfo.flags =
            updateAfterBind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
        descriptorLayoutCreateInfo.pNext = &bindingFlagsInfo;
        descriptorLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setBindingBuilder.bindings.size());
        descriptorLayoutCreateInfo.pBindings = setBindingBuilder.bindings.data();

        if (vkCreateDescriptorSetLayout(device->vkDevice(), &descriptorLayoutCreateInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS) {
            descriptorTableCerr() << "failed to create descriptor layout" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (VkDescriptorSetLayoutBinding &binding : setBindingBuilder.bindings) {
            auto it = descriptorTypeCount.find(binding.descriptorType);
            if (it != descriptorTypeCount.end()) {
                it->second += binding.descriptorCount;
            } else {
                descriptorTypeCount.emplace(binding.descriptorType, binding.descriptorCount);
            }
        }
    }

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets(descriptorSetLayouts.size(), VK_NULL_HANDLE);

    std::vector<VkDescriptorSetLayout> regularSetLayouts;
    std::vector<uint32_t> regularSetIndices;
    for (uint32_t i = 0; i < static_cast<uint32_t>(pushDescriptorSets.size()); i++) {
        if (!pushDescriptorSets[i]) {
            regularSetLayouts.push_back(descriptorSetLayouts[i]);
            regularSetIndices.push_back(i);
        }
    }

    if (!regularSetLayouts.empty()) {
        std::vector<VkDescriptorPoolSize> poolSizes;
        for (auto &[type, cnt] : descriptorTypeCount) {
            poolSizes.push_back({
                .type = type,
                .descriptorCount = cnt,
            });
        }

        VkDescriptorPoolCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.flags = hasUpdateAfterBindSet ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
        createInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        createInfo.pPoolSizes = poolSizes.data();
        createInfo.maxSets = static_cast<uint32_t>(regularSetLayouts.size());

        if (vkCreateDescriptorPool(device->vkDevice(), &createInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            descriptorTableCerr() << "failed to create descriptor pool" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<VkDescriptorSet> regularDescriptorSets(regularSetLayouts.size());
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(regularSetLayouts.size());
        allocInfo.pSetLayouts = regularSetLayouts.data();

        if (vkAllocateDescriptorSets(device->vkDevice(), &allocInfo, regularDescriptorSets.data()) != VK_SUCCESS) {
            descriptorTableCerr() << "failed to create descriptor set" << std::endl;
            exit(1);
        }

        for (uint32_t i = 0; i < static_cast<uint32_t>(regularSetIndices.size()); i++) {
            descriptorSets[regularSetIndices[i]] = regularDescriptorSets[i];
        }
    }

    std::vector<std::vector<VkDescriptorType>> descriptorTypes;
    for (int i = 0; i < static_cast<int>(setBuilders.setBindingBuilders.size()); i++) {
        const auto &bindings = setBuilders.setBindingBuilders[i].bindings;
        uint32_t maxBinding = 0;
        for (const auto &binding : bindings) {
            if (binding.binding > maxBinding) { maxBinding = binding.binding; }
        }
        std::vector<VkDescriptorType> &types = descriptorTypes.emplace_back();
        types.resize(maxBinding + 1, VK_DESCRIPTOR_TYPE_MAX_ENUM);
        for (const auto &binding : bindings) { types[binding.binding] = binding.descriptorType; }
    }

    std::vector<VkPushConstantRange> normalizedPushConstantRanges = normalizePushConstantRanges(pushConstantRanges);
    return std::make_shared<vk::DescriptorTable>(device, descriptorPool, descriptorSetLayouts, descriptorSets,
                                               descriptorTypes, pushDescriptorSets, normalizedPushConstantRanges);
}
