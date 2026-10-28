#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/vertex.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace vk {
class Device;
class RenderPass;
class DescriptorTable;
class GraphicsPipelineBuilder;
class Shader;

class DynamicGraphicsPipelineBuilder;

class DynamicGraphicsPipeline : public SharedObject<DynamicGraphicsPipeline> {
    friend DynamicGraphicsPipelineBuilder;

  public:
    DynamicGraphicsPipeline(std::shared_ptr<Device> device, VkPipeline pipeline);
    ~DynamicGraphicsPipeline();

    VkPipeline vkPipeline();

  private:
    std::shared_ptr<Device> device_;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class DynamicGraphicsPipelineBuilder {
  public:
    struct ShaderStageBuilder {
        DynamicGraphicsPipelineBuilder &parent;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;

        ShaderStageBuilder(DynamicGraphicsPipelineBuilder &parent);

        ShaderStageBuilder &defineShaderStage(VkPipelineShaderStageCreateInfo shaderStageCreateInfo);
        ShaderStageBuilder &defineShaderStage(VkShaderModule shaderModule, VkShaderStageFlagBits shaderStage);
        ShaderStageBuilder &defineShaderStage(std::shared_ptr<Shader> shader, VkShaderStageFlagBits shaderStage);
        DynamicGraphicsPipelineBuilder &endShaderStage();
    };

  public:
    DynamicGraphicsPipelineBuilder(int numColorAttachments);

    DynamicGraphicsPipelineBuilder &defineRenderPass(std::shared_ptr<RenderPass> renderPass, uint32_t subpassIndex);

    ShaderStageBuilder &beginShaderStage();

    template <typename T>
    DynamicGraphicsPipelineBuilder &defineVertexInputState();

    DynamicGraphicsPipelineBuilder &defineVertexInputState(const VertexLayoutInfo &vertexLayoutInfo);

    DynamicGraphicsPipelineBuilder &definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable);

    DynamicGraphicsPipelineBuilder &defineInputAssemblyState(VkPrimitiveTopology topology);

    DynamicGraphicsPipelineBuilder &defineColorBlendAttachmentState(uint32_t attachment,
                                                                    VkPipelineColorBlendAttachmentState state);

    std::shared_ptr<vk::DynamicGraphicsPipeline> build(std::shared_ptr<Device> device);

  private:
    constexpr static VkDynamicState dynamicState_[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,

        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,

        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_LINE_WIDTH,

        VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
        VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
        VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,

        VK_DYNAMIC_STATE_LOGIC_OP_EXT,
        VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,

    };

    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = sizeof(dynamicState_) / sizeof(*dynamicState_),
        .pDynamicStates = dynamicState_,
    };

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineViewportStateCreateInfo viewportStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
    };

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates_;

    VkPipelineColorBlendStateCreateInfo colorBlendState_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 0,
        .pAttachments = nullptr,
    };

    VkPipelineDepthStencilStateCreateInfo depthStencilState_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthBoundsTestEnable = VK_FALSE,
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    VkPipelineVertexInputStateCreateInfo vertexInputState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    ShaderStageBuilder shaderStageBuilder_;

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    uint32_t subpassIndex_ = 0;
};

template <typename T>
DynamicGraphicsPipelineBuilder &DynamicGraphicsPipelineBuilder::defineVertexInputState() {
    VertexLayoutInfo &vertexLayoutInfo = Vertex::vertexLayoutInfo<T>();
    return defineVertexInputState(vertexLayoutInfo);
}

inline DynamicGraphicsPipelineBuilder &
DynamicGraphicsPipelineBuilder::defineVertexInputState(const VertexLayoutInfo &vertexLayoutInfo) {
    vertexInputStateCreateInfo_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCreateInfo_.vertexBindingDescriptionCount = 1;
    vertexInputStateCreateInfo_.pVertexBindingDescriptions = &vertexLayoutInfo.bindingDescription;
    vertexInputStateCreateInfo_.vertexAttributeDescriptionCount = vertexLayoutInfo.attributeDescriptions.size();
    vertexInputStateCreateInfo_.pVertexAttributeDescriptions = vertexLayoutInfo.attributeDescriptions.data();

    return *this;
}

template <>
DynamicGraphicsPipelineBuilder &DynamicGraphicsPipelineBuilder::defineVertexInputState<void>();
}
