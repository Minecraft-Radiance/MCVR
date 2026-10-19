#include "core/vulkan/pipeline.hpp"

#include "core/vulkan/descriptor.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/render_pass.hpp"
#include "core/vulkan/shader.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

std::ostream &graphicsPipelineCerr() {
    return std::cerr << "[GraphicsPipeline] ";
}

vk::GraphicsPipeline::GraphicsPipeline(std::shared_ptr<Device> device, VkPipeline pipeline)
    : device_(device), pipeline_(pipeline) {}

vk::GraphicsPipeline::~GraphicsPipeline() {
    vkDestroyPipeline(device_->vkDevice(), pipeline_, nullptr);
}

VkPipeline &vk::GraphicsPipeline::vkPipeline() {
    return pipeline_;
}

vk::RayTracingPipeline::RayTracingPipeline(std::shared_ptr<Device> device, VkPipeline pipeline)
    : device_(device), pipeline_(pipeline) {}

vk::RayTracingPipeline::~RayTracingPipeline() {
    vkDestroyPipeline(device_->vkDevice(), pipeline_, nullptr);
}

VkPipeline &vk::RayTracingPipeline::vkPipeline() {
    return pipeline_;
}

vk::ComputePipeline::ComputePipeline(std::shared_ptr<Device> device, VkPipeline pipeline)
    : device_(device), pipeline_(pipeline) {}

vk::ComputePipeline::~ComputePipeline() {
    vkDestroyPipeline(device_->vkDevice(), pipeline_, nullptr);
}

VkPipeline &vk::ComputePipeline::vkPipeline() {
    return pipeline_;
}

vk::GraphicsPipelineBuilder::ShaderStageBuilder::ShaderStageBuilder(GraphicsPipelineBuilder &parent) : parent(parent) {}

vk::GraphicsPipelineBuilder::ShaderStageBuilder &vk::GraphicsPipelineBuilder::ShaderStageBuilder::defineShaderStage(
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo) {
    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    return *this;
}

vk::GraphicsPipelineBuilder::ShaderStageBuilder &
vk::GraphicsPipelineBuilder::ShaderStageBuilder::defineShaderStage(VkShaderModule shaderModule,
                                                                   VkShaderStageFlagBits shaderStage) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = shaderStage;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";

    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    return *this;
}

vk::GraphicsPipelineBuilder::ShaderStageBuilder &
vk::GraphicsPipelineBuilder::ShaderStageBuilder::defineShaderStage(std::shared_ptr<Shader> shader,
                                                                   VkShaderStageFlagBits shaderStage) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = shaderStage;
    shaderStageCreateInfo.module = shader->vkShaderModule();
    shaderStageCreateInfo.pName = "main";

    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    return *this;
}

vk::GraphicsPipelineBuilder &vk::GraphicsPipelineBuilder::ShaderStageBuilder::endShaderStage() {
    return parent;
}

vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder::ColorBlendAttachmentStateBuilder(
    vk::GraphicsPipelineBuilder &parent)
    : parent(parent) {}

vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder &
vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder::defineColorBlendAttachmentState(
    VkPipelineColorBlendAttachmentState colorBlendAttachmentState) {
    colorBlendAttachmentStates.push_back(colorBlendAttachmentState);
    return *this;
}

vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder &
vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder::defineDefaultColorBlendAttachmentState() {
    VkPipelineColorBlendAttachmentState colorBlendAttachmentState = {
        colorBlendAttachmentState.blendEnable = VK_FALSE,
        colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        colorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD,
        colorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        colorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        colorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD,
        colorBlendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    colorBlendAttachmentStates.push_back(colorBlendAttachmentState);
    return *this;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder::endColorBlendAttachmentState() {
    parent.colorBlendState_.attachmentCount = colorBlendAttachmentStates.size();
    parent.colorBlendState_.pAttachments = colorBlendAttachmentStates.data();
    return parent;
}

vk::GraphicsPipelineBuilder::GraphicsPipelineBuilder()
    : shaderStageBuilder_(*this), colorBlendAttachmentStateBuilder_(*this) {}

vk::GraphicsPipelineBuilder &vk::GraphicsPipelineBuilder::defineRenderPass(std::shared_ptr<RenderPass> renderPass,
                                                                           uint32_t subpassIndex) {
    renderPass_ = renderPass->vkRenderPass();
    subpassIndex_ = subpassIndex;
    return *this;
}

vk::GraphicsPipelineBuilder::ShaderStageBuilder &vk::GraphicsPipelineBuilder::beginShaderStage() {
    return shaderStageBuilder_;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::defineVertexInputState(VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo) {
    vertexInputStateCreateInfo_ = vertexInputStateCreateInfo;
    return *this;
}

template <>
vk::GraphicsPipelineBuilder &vk::GraphicsPipelineBuilder::GraphicsPipelineBuilder::defineVertexInputState<void>() {
    vertexInputStateCreateInfo_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCreateInfo_.vertexBindingDescriptionCount = 0;
    vertexInputStateCreateInfo_.pVertexBindingDescriptions = nullptr;
    vertexInputStateCreateInfo_.vertexAttributeDescriptionCount = 0;
    vertexInputStateCreateInfo_.pVertexAttributeDescriptions = nullptr;

    return *this;
}

vk::GraphicsPipelineBuilder &vk::GraphicsPipelineBuilder::defineInputAssemblyState(
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo) {
    inputAssemblyStateCreateInfo_ = inputAssemblyStateCreateInfo;
    return *this;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::defineViewportState(VkPipelineViewportStateCreateInfo viewportStateCreateInfo) {
    viewportStateCreateInfo_ = viewportStateCreateInfo;
    return *this;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::defineViewportScissorState(ViewPortAndScissor viewPortAndScissor) {
    viewPortAndScissor_ = viewPortAndScissor;
    VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewPortAndScissor_.viewport,
        .scissorCount = 1,
        .pScissors = &viewPortAndScissor_.scissor,
    };
    viewportStateCreateInfo_ = viewportStateCreateInfo;
    return *this;
}

vk::GraphicsPipelineBuilder &vk::GraphicsPipelineBuilder::defineRasterizationState(
    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo) {
    rasterizationStateCreateInfo_ = rasterizationStateCreateInfo;
    return *this;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::defineMultisampleState(VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo) {
    multisampleStateCreateInfo_ = multisampleStateCreateInfo;
    return *this;
}

vk::GraphicsPipelineBuilder::ColorBlendAttachmentStateBuilder &
vk::GraphicsPipelineBuilder::beginColorBlendAttachmentState() {
    return colorBlendAttachmentStateBuilder_;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::defineColorBlendState(VkPipelineColorBlendStateCreateInfo colorBlendState) {
    colorBlendState_ = colorBlendState;
    return *this;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable) {
    pipelineLayout_ = descriptorTable->vkPipelineLayout();
    return *this;
}

vk::GraphicsPipelineBuilder &vk::GraphicsPipelineBuilder::definePipelineLayout(VkPipelineLayout pipelineLayout) {
    pipelineLayout_ = pipelineLayout;
    return *this;
}

vk::GraphicsPipelineBuilder &
vk::GraphicsPipelineBuilder::defineDepthStencilState(vk::GraphicsPipelineBuilder::DepthStencilState depthStencilState) {
    depthStencilState_.depthTestEnable = depthStencilState.depthTestEnable;
    depthStencilState_.depthWriteEnable = depthStencilState.depthWriteEnable;
    depthStencilState_.depthCompareOp = depthStencilState.depthCompareOp;
    depthStencilState_.depthBoundsTestEnable = depthStencilState.depthBoundsTestEnable;
    depthStencilState_.stencilTestEnable = depthStencilState.stencilTestEnable;
    depthStencilState_.front = depthStencilState.front;
    depthStencilState_.back = depthStencilState.back;
    depthStencilState_.minDepthBounds = depthStencilState.minDepthBounds;
    depthStencilState_.maxDepthBounds = depthStencilState.maxDepthBounds;
    return *this;
}

std::shared_ptr<vk::GraphicsPipeline> vk::GraphicsPipelineBuilder::build(std::shared_ptr<Device> device) {
    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = shaderStageBuilder_.shaderStageCreateInfos.size();
    pipelineCreateInfo.pStages = shaderStageBuilder_.shaderStageCreateInfos.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo_;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo_;
    pipelineCreateInfo.pViewportState = &viewportStateCreateInfo_;
    pipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo_;
    pipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo_;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState_;
    pipelineCreateInfo.pColorBlendState = &colorBlendState_;
    pipelineCreateInfo.layout = pipelineLayout_;
    pipelineCreateInfo.renderPass = renderPass_;
    pipelineCreateInfo.subpass = subpassIndex_;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = -1;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device->vkDevice(), device->vkPipelineCache(), 1, &pipelineCreateInfo, nullptr, &pipeline) !=
        VK_SUCCESS) {
        graphicsPipelineCerr() << "failed to create graphics pipeline" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::make_shared<GraphicsPipeline>(device, pipeline);
}

vk::RayTracingPipelineBuilder::ShaderStageBuilder::ShaderStageBuilder(RayTracingPipelineBuilder &parent)
    : parent(parent) {}

vk::RayTracingPipelineBuilder::ShaderStageBuilder &
vk::RayTracingPipelineBuilder::ShaderStageBuilder::defineShaderStage(std::shared_ptr<Shader> shader,
                                                                     VkShaderStageFlagBits shaderStage) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = shaderStage;
    shaderStageCreateInfo.module = shader->vkShaderModule();
    shaderStageCreateInfo.pName = "main";

    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    shaders.push_back(std::move(shader));
    return *this;
}

vk::RayTracingPipelineBuilder &vk::RayTracingPipelineBuilder::ShaderStageBuilder::endShaderStage() {
    return parent;
}

vk::RayTracingPipelineBuilder::ShaderGroupBuilder::ShaderGroupBuilder(RayTracingPipelineBuilder &parent)
    : parent(parent) {}

vk::RayTracingPipelineBuilder::ShaderGroupBuilder &
vk::RayTracingPipelineBuilder::ShaderGroupBuilder::defineShaderGroup(VkRayTracingShaderGroupTypeKHR type,
                                                                     uint32_t generalShader,
                                                                     uint32_t closestHitShader,
                                                                     uint32_t anyHitShader,
                                                                     uint32_t intersectionShader) {
    VkRayTracingShaderGroupCreateInfoKHR shaderGroupCreateInfo{};
    shaderGroupCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shaderGroupCreateInfo.type = type;
    shaderGroupCreateInfo.generalShader = generalShader;
    shaderGroupCreateInfo.closestHitShader = closestHitShader;
    shaderGroupCreateInfo.anyHitShader = anyHitShader;
    shaderGroupCreateInfo.intersectionShader = intersectionShader;

    shaderGroupCreateInfos.push_back(shaderGroupCreateInfo);
    return *this;
}

vk::RayTracingPipelineBuilder &vk::RayTracingPipelineBuilder::ShaderGroupBuilder::endShaderGroup() {
    return parent;
}

vk::RayTracingPipelineBuilder::RayTracingPipelineBuilder() : shaderStageBuilder_(*this), shaderGroupBuilder_(*this) {}

vk::RayTracingPipelineBuilder::ShaderStageBuilder &vk::RayTracingPipelineBuilder::beginShaderStage() {
    return shaderStageBuilder_;
}

vk::RayTracingPipelineBuilder::ShaderGroupBuilder &vk::RayTracingPipelineBuilder::beginShaderGroup() {
    return shaderGroupBuilder_;
}

vk::RayTracingPipelineBuilder &
vk::RayTracingPipelineBuilder::definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable) {
    pipelineLayout_ = descriptorTable->vkPipelineLayout();
    return *this;
}

vk::RayTracingPipelineBuilder &vk::RayTracingPipelineBuilder::defineMaxRayRecursionDepth(uint32_t depth) {
    if (depth == 0) { throw std::runtime_error("ray tracing recursion depth must be at least one"); }
    maxRayRecursionDepth_ = depth;
    return *this;
}

std::shared_ptr<vk::RayTracingPipeline> vk::RayTracingPipelineBuilder::build(std::shared_ptr<Device> device) {
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = shaderStageBuilder_.shaderStageCreateInfos.size();
    pipelineInfo.pStages = shaderStageBuilder_.shaderStageCreateInfos.data();
    pipelineInfo.groupCount = shaderGroupBuilder_.shaderGroupCreateInfos.size();
    pipelineInfo.pGroups = shaderGroupBuilder_.shaderGroupCreateInfos.data();
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.maxPipelineRayRecursionDepth = maxRayRecursionDepth_;
#ifndef __APPLE__
    pipelineInfo.flags = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_AABBS_BIT_KHR;
#endif

    VkPipeline rayTracingPipeline;
    if (vkCreateRayTracingPipelinesKHR(device->vkDevice(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                       &rayTracingPipeline) != VK_SUCCESS) {
        std::cerr << "Cannot build ray tracing pipeline" << std::endl;
        exit(EXIT_FAILURE);
    }

    return RayTracingPipeline::create(device, rayTracingPipeline);
}

vk::ComputePipelineBuilder &vk::ComputePipelineBuilder::defineShader(std::shared_ptr<vk::Shader> shader) {
    shader_ = std::move(shader);
    return *this;
}

vk::ComputePipelineBuilder &
vk::ComputePipelineBuilder::definePipelineLayout(std::shared_ptr<vk::DescriptorTable> descriptorTable) {
    pipelineLayout_ = descriptorTable->vkPipelineLayout();
    return *this;
}

std::shared_ptr<vk::ComputePipeline> vk::ComputePipelineBuilder::build(std::shared_ptr<vk::Device> device) {
    std::vector<VkSpecializationMapEntry> specializationEntries;
    specializationEntries.reserve(specializationConstants_.size());
    for (uint32_t i = 0; i < specializationConstants_.size(); ++i) {
        specializationEntries.push_back({i, i * uint32_t(sizeof(uint32_t)), sizeof(uint32_t)});
    }
    const VkSpecializationInfo specializationInfo{
        .mapEntryCount = static_cast<uint32_t>(specializationEntries.size()),
        .pMapEntries = specializationEntries.data(),
        .dataSize = specializationConstants_.size() * sizeof(uint32_t),
        .pData = specializationConstants_.data(),
    };
    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineCreateInfo.stage.module = shader_->vkShaderModule();
    computePipelineCreateInfo.stage.pName = "main";
    computePipelineCreateInfo.stage.pSpecializationInfo =
        specializationConstants_.empty() ? nullptr : &specializationInfo;
    computePipelineCreateInfo.layout = pipelineLayout_;

    VkPipeline compPipeline;
    if (vkCreateComputePipelines(device->vkDevice(), device->vkPipelineCache(), 1, &computePipelineCreateInfo, nullptr,
                                 &compPipeline) != VK_SUCCESS) {
        std::cerr << "Cannot build compute pipeline" << std::endl;
        exit(EXIT_FAILURE);
    }

    return ComputePipeline::create(device, compPipeline);
}

vk::ComputePipelineBuilder &vk::ComputePipelineBuilder::defineSpecializationConstants(std::vector<uint32_t> values) {
    specializationConstants_ = std::move(values);
    return *this;
}
