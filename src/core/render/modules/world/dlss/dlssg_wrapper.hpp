/*
 * Copyright (c) 2026 Radiance
 *
 * This file is licensed under the GNU General Public License
 * as published by the Free Software Foundation; either version 3 of the License,
 * or (at your option) any later version.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"

#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_defs_dlssg.h"

#include <glm/glm.hpp>
#include <array>

namespace vk {
class Device;
class CommandPool;
class CommandBuffer;
class DeviceLocalImage;
class SwapchainImage;
class Image;
}; // namespace vk

class NgxContext;

class DlssFG : public SharedObject<DlssFG> {
  public:
    DlssFG() = default;
    ~DlssFG();

    enum FGResource {
        RESOURCE_BACKBUFFER = 0,
        RESOURCE_DEPTH,
        RESOURCE_MVECS,
        RESOURCE_OUTPUT_INTERP,

        RESOURCE_NUM
    };

    void setResource(FGResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image, bool readWrite = false);
    void setResource(FGResource resourceId, std::shared_ptr<vk::SwapchainImage> image, bool readWrite = false);
    void resetResource(FGResource resourceId);

    NVSDK_NGX_Result evaluate(std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                              const glm::mat4 &viewMatrix,
                              const glm::mat4 &projMatrix,
                              const glm::mat4 &prevViewMatrix,
                              const glm::mat4 &prevProjMatrix,
                              glm::vec2 jitter,
                              glm::vec3 cameraPos,
                              float cameraNear,
                              float cameraFar,
                              float cameraFOV,
                              float cameraAspectRatio,
                              bool reset = false);

    void deinit();

  private:
    friend class NgxContext;
    NVSDK_NGX_Result init(std::shared_ptr<vk::Device> device,
                          std::shared_ptr<vk::CommandPool> cmdPool,
                          NVSDK_NGX_Parameter *ngxParams,
                          uint32_t width,
                          uint32_t height,
                          VkFormat backbufferFormat);

    DlssFG(const DlssFG &) = delete;
    DlssFG(const DlssFG &&) = delete;
    DlssFG &operator=(const DlssFG &) = delete;
    DlssFG &operator=(const DlssFG &&) = delete;

    std::shared_ptr<vk::Device> m_device;
    NVSDK_NGX_Parameter *m_ngxParams = nullptr;
    NVSDK_NGX_Handle *m_dlssgHandle = nullptr;
    VkExtent2D m_size;
    std::array<NVSDK_NGX_Resource_VK, RESOURCE_NUM> m_resources;
};
