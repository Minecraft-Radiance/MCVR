#include "core/vulkan/dynamic_pipeline.hpp"

#include "core/vulkan/descriptor.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/render_pass.hpp"
#include "core/vulkan/shader.hpp"

#include <iostream>
#include <vector>

std::ostream &dynamicGraphicsPipelineCerr() {
    return std::cerr << "[GraphicsPipeline] ";
}

vk::DynamicGraphicsPipeline::DynamicGraphicsPipeline(std::shared_ptr<Device> device, VkPipeline pipeline)
    : device_(device), pipeline_(pipeline) {}

vk::DynamicGraphicsPipeline::~DynamicGraphicsPipeline() {
    vkDestroyPipeline(device_->vkDevice(), pipeline_, nullptr);
}

VkPipeline vk::DynamicGraphicsPipeline::vkPipeline() {
    return pipeline_;
}

vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder::ShaderStageBuilder(DynamicGraphicsPipelineBuilder &parent)
    : parent(parent) {}

vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder &
vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder::defineShaderStage(
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo) {
    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    return *this;
}

vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder &
vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder::defineShaderStage(VkShaderModule shaderModule,
                                                                          VkShaderStageFlagBits shaderStage) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = shaderStage;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";

    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    return *this;
}

vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder &
vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder::defineShaderStage(std::shared_ptr<Shader> shader,
                                                                          VkShaderStageFlagBits shaderStage) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = shaderStage;
    shaderStageCreateInfo.module = shader->vkShaderModule();
    shaderStageCreateInfo.pName = "main";

    shaderStageCreateInfos.push_back(shaderStageCreateInfo);
    return *this;
}

vk::DynamicGraphicsPipelineBuilder &vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder::endShaderStage() {
    return parent;
}

vk::DynamicGraphicsPipelineBuilder::DynamicGraphicsPipelineBuilder(int numColorAttachments)
    : shaderStageBuilder_(*this) {
    colorBlendAttachmentStates_.resize(numColorAttachments);
    for (int i = 0; i < numColorAttachments; i++) {
        colorBlendAttachmentStates_[i] = {
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
    }
    colorBlendState_.attachmentCount = numColorAttachments;
    colorBlendState_.pAttachments = colorBlendAttachmentStates_.data();
}

vk::DynamicGraphicsPipelineBuilder &
vk::DynamicGraphicsPipelineBuilder::defineRenderPass(std::shared_ptr<RenderPass> renderPass, uint32_t subpassIndex) {
    renderPass_ = renderPass->vkRenderPass();
    subpassIndex_ = subpassIndex;
    return *this;
}

vk::DynamicGraphicsPipelineBuilder::ShaderStageBuilder &vk::DynamicGraphicsPipelineBuilder::beginShaderStage() {
    return shaderStageBuilder_;
}

vk::DynamicGraphicsPipelineBuilder &
vk::DynamicGraphicsPipelineBuilder::definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable) {
    pipelineLayout_ = descriptorTable->vkPipelineLayout();
    return *this;
}

vk::DynamicGraphicsPipelineBuilder &
vk::DynamicGraphicsPipelineBuilder::defineInputAssemblyState(VkPrimitiveTopology topology) {
    inputAssemblyStateCreateInfo_.topology = topology;
    return *this;
}

vk::DynamicGraphicsPipelineBuilder &vk::DynamicGraphicsPipelineBuilder::defineColorBlendAttachmentState(
    uint32_t attachment, VkPipelineColorBlendAttachmentState state) {
    colorBlendAttachmentStates_.at(attachment) = state;
    return *this;
}

std::shared_ptr<vk::DynamicGraphicsPipeline> vk::DynamicGraphicsPipelineBuilder::build(std::shared_ptr<Device> device) {
    std::vector<VkDynamicState> filteredDynamicStates;
    for (uint32_t i = 0; i < sizeof(dynamicState_) / sizeof(*dynamicState_); i++) {
        VkDynamicState state = dynamicState_[i];
        if ((state == VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT ||
             state == VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT ||
             state == VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT) && !device->hasDynamicColorBlendState()) {
            continue;
        }
        if ((state == VK_DYNAMIC_STATE_LOGIC_OP_EXT || state == VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT) &&
            !device->hasExtendedDynamicState2LogicOp()) {
            continue;
        }
        filteredDynamicStates.push_back(state);
    }

    VkPipelineDynamicStateCreateInfo filteredDynamicStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(filteredDynamicStates.size()),
        .pDynamicStates = filteredDynamicStates.data(),
    };

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
    pipelineCreateInfo.pDynamicState = &filteredDynamicStateCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout_;
    pipelineCreateInfo.renderPass = renderPass_;
    pipelineCreateInfo.subpass = subpassIndex_;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = -1;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device->vkDevice(), device->vkPipelineCache(), 1, &pipelineCreateInfo, nullptr, &pipeline) !=
        VK_SUCCESS) {
        dynamicGraphicsPipelineCerr() << "failed to create graphics pipeline" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::make_shared<DynamicGraphicsPipeline>(device, pipeline);
}
