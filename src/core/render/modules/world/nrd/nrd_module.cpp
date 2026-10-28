#include "nrd_module.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr float NRD_DENOISING_RANGE = 58953.0f;

struct NrdModePushConstants {
    uint32_t shEnabled;
    uint32_t separateDirectEnabled;
    uint32_t frameIndex;
    uint32_t nrdMode;
};
struct NrdLightingPushConstants {
    uint32_t hasPrevious;
    float jitterDeltaX, jitterDeltaY;
};
struct NrdResolvePushConstants {
    uint32_t hasPrevious;
    float jitterDeltaX;
    float jitterDeltaY;
    float padding = 0;
    glm::vec4 currentRayX, currentRayY, currentRayOrigin;
    glm::vec4 previousRayX, previousRayY, previousRayOrigin;
};
static_assert(sizeof(NrdResolvePushConstants) == 112);
}

nrd::ReblurSettings NrdModule::makeDefaultReblurSettings() {
    nrd::ReblurSettings settings{};
    settings.checkerboardMode = nrd::CheckerboardMode::BLACK;
    settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    settings.maxAccumulatedFrameNum = 30;
    settings.maxFastAccumulatedFrameNum = 6;
    settings.maxStabilizedFrameNum = 30;
    settings.responsiveAccumulationSettings.roughnessThreshold = 0.0f;
    settings.responsiveAccumulationSettings.minAccumulatedFrameNum = 0;
    settings.diffusePrepassBlurRadius = 0.0f;
    settings.specularPrepassBlurRadius = 20.0f;
    settings.minMaterialForDiffuse = 0.0f;
    settings.minMaterialForSpecular = 0.0f;
    settings.enableAntiFirefly = true;
    settings.enableLowRoughnessSpecularStabilization = true;
    settings.fireflySuppressorMinRelativeScale = 1.0f;
    settings.usePrepassOnlyForSpecularMotionEstimation = true;
    settings.fastHistoryClampingSigmaScale = 2.0f;
    settings.antilagSettings.luminanceSigmaScale = 4.0f;
    settings.antilagSettings.luminanceSensitivity = 3.0f;
    return settings;
}

nrd::RelaxSettings NrdModule::makeDefaultRelaxSettings() {
    nrd::RelaxSettings settings = {};
    settings.antilagSettings.accelerationAmount = 0.3f;
    settings.antilagSettings.resetAmount = 0.5f;
    settings.diffuseMaxAccumulatedFrameNum = 30;
    settings.specularMaxAccumulatedFrameNum = 12;
    settings.diffuseMaxFastAccumulatedFrameNum = 6;
    settings.specularMaxFastAccumulatedFrameNum = 4;
    settings.historyFixFrameNum = 3;
    settings.fastHistoryClampingSigmaScale = 1.35f;
    settings.diffusePrepassBlurRadius = 0.0f;
    settings.specularPrepassBlurRadius = 0.0f;
    settings.minHitDistanceWeight = 0.05f;
    settings.diffusePhiLuminance = 2.5f;
    settings.specularPhiLuminance = 1.5f;
    settings.diffuseMinLuminanceWeight = 0.05f;
    settings.specularMinLuminanceWeight = 0.02f;
    settings.lobeAngleFraction = 0.35f;
    settings.roughnessFraction = 0.2f;
    settings.specularVarianceBoost = 0.15f;
    settings.atrousIterationNum = 3;
    settings.depthThreshold = 0.003f;
    settings.minMaterialForDiffuse = 0.0f;
    settings.minMaterialForSpecular = 0.0f;
    settings.enableAntiFirefly = true;
    settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    return settings;
}

std::shared_ptr<vk::DeviceLocalImage> NrdModule::createInputImage(const std::shared_ptr<Framework> &framework,
                                                                  uint32_t width,
                                                                  uint32_t height,
                                                                  VkFormat format) {
    return vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, width, height, 1, format,
                                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                            VK_IMAGE_USAGE_SAMPLED_BIT);
}

bool NrdModule::parseBool(const std::string &value) {
    return value == "render_pipeline.true" || value == "1" || value == "true" || value == "True" || value == "TRUE";
}

uint32_t NrdModule::parseUint(const std::string &value, uint32_t fallback) {
    try {
        long long parsed = std::stoll(value);
        parsed = std::clamp(parsed, 0LL, static_cast<long long>(std::numeric_limits<uint32_t>::max()));
        return static_cast<uint32_t>(parsed);
    } catch (...) { return fallback; }
}

float NrdModule::parseFloat(const std::string &value, float fallback) {
    try {
        return std::stof(value);
    } catch (...) { return fallback; }
}

nrd::HitDistanceReconstructionMode
NrdModule::parseHitDistanceReconstructionMode(const std::string &value, nrd::HitDistanceReconstructionMode fallback) {
    if (value == "off") return nrd::HitDistanceReconstructionMode::OFF;
    if (value == "3x3") return nrd::HitDistanceReconstructionMode::AREA_3X3;
    if (value == "5x5") return nrd::HitDistanceReconstructionMode::AREA_5X5;
    return fallback;
}

void NrdModule::updateReblurSettings() {
    auto &settings = reblurSettings_;

    const bool checkerboard = nrdMode_ == ShaderPackLoader::NrdMode::Checkerboard ||
                              nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard;
    settings.checkerboardMode = checkerboard ? nrd::CheckerboardMode::BLACK : nrd::CheckerboardMode::OFF;
    settings.hitDistanceReconstructionMode =
        checkerboard ? nrd::HitDistanceReconstructionMode::OFF : configuredReblurHitDistanceReconstructionMode_;
    settings.enableAntiFirefly = true;
    settings.enableLowRoughnessSpecularStabilization = true;
    settings.fireflySuppressorMinRelativeScale = 1.0f;
    settings.fastHistoryClampingSigmaScale = 2.0f;
    settings.antilagSettings.luminanceSigmaScale = 4.0f;
    settings.antilagSettings.luminanceSensitivity = 3.0f;

    settings.maxAccumulatedFrameNum = std::min(settings.maxAccumulatedFrameNum, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
    settings.maxFastAccumulatedFrameNum =
        std::min(settings.maxFastAccumulatedFrameNum, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
    settings.maxStabilizedFrameNum = std::min(settings.maxStabilizedFrameNum, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
    settings.historyFixFrameNum = std::min(settings.historyFixFrameNum, 3u);
    settings.historyFixFrameNum = settings.maxFastAccumulatedFrameNum == 0 ?
                                      0u :
                                      std::min(settings.historyFixFrameNum, settings.maxFastAccumulatedFrameNum - 1u);
    settings.historyFixBasePixelStride = std::max(settings.historyFixBasePixelStride, 1u);
    settings.historyFixAlternatePixelStride = std::max(settings.historyFixAlternatePixelStride, 1u);
    settings.fastHistoryClampingSigmaScale = std::clamp(settings.fastHistoryClampingSigmaScale, 1.0f, 3.0f);
    settings.diffusePrepassBlurRadius = std::max(settings.diffusePrepassBlurRadius, 0.0f);
    settings.specularPrepassBlurRadius = std::max(settings.specularPrepassBlurRadius, 0.0f);
    settings.minHitDistanceWeight = std::clamp(settings.minHitDistanceWeight, 0.0001f, 0.2f);
    settings.minBlurRadius = std::max(settings.minBlurRadius, 0.0f);
    settings.maxBlurRadius = std::max(settings.maxBlurRadius, settings.minBlurRadius);
    settings.lobeAngleFraction = std::clamp(settings.lobeAngleFraction, 0.0f, 1.0f);
    settings.roughnessFraction = std::clamp(settings.roughnessFraction, 0.0f, 1.0f);
    settings.planeDistanceSensitivity = std::max(settings.planeDistanceSensitivity, 0.0f);
    settings.specularProbabilityThresholdsForMvModification[0] =
        std::clamp(settings.specularProbabilityThresholdsForMvModification[0], 0.0f, 1.0f);
    settings.specularProbabilityThresholdsForMvModification[1] =
        std::clamp(settings.specularProbabilityThresholdsForMvModification[1], 0.0f, 1.0f);
    if (settings.specularProbabilityThresholdsForMvModification[1] <
        settings.specularProbabilityThresholdsForMvModification[0]) {
        settings.specularProbabilityThresholdsForMvModification[1] =
            settings.specularProbabilityThresholdsForMvModification[0];
    }
    settings.fireflySuppressorMinRelativeScale = std::clamp(settings.fireflySuppressorMinRelativeScale, 1.0f, 3.0f);
    settings.enableAntiFirefly = true;
    settings.minMaterialForDiffuse = std::max(settings.minMaterialForDiffuse, 0.0f);
    settings.minMaterialForSpecular = std::max(settings.minMaterialForSpecular, 0.0f);
    settings.antilagSettings.luminanceSigmaScale = std::clamp(settings.antilagSettings.luminanceSigmaScale, 1.0f, 5.0f);
    settings.antilagSettings.luminanceSensitivity =
        std::clamp(settings.antilagSettings.luminanceSensitivity, 1.0f, 5.0f);
    settings.responsiveAccumulationSettings.roughnessThreshold = 0.0f;
    settings.responsiveAccumulationSettings.minAccumulatedFrameNum = 0;

    settings.usePrepassOnlyForSpecularMotionEstimation = nrdMode_ != ShaderPackLoader::NrdMode::HalfRateCheckerboard;
    settings.enableHalfRateTransmission = nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard;
}

void NrdModule::updateRelaxSettings() {
    auto &settings = relaxSettings_;
    settings.antilagSettings.accelerationAmount = std::clamp(settings.antilagSettings.accelerationAmount, 0.0f, 1.0f);
    settings.antilagSettings.spatialSigmaScale = std::max(settings.antilagSettings.spatialSigmaScale, 0.0f);
    settings.antilagSettings.temporalSigmaScale = std::max(settings.antilagSettings.temporalSigmaScale, 0.0f);
    settings.antilagSettings.resetAmount = std::clamp(settings.antilagSettings.resetAmount, 0.0f, 1.0f);
    settings.diffuseMaxAccumulatedFrameNum =
        std::min(settings.diffuseMaxAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    settings.specularMaxAccumulatedFrameNum =
        std::min(settings.specularMaxAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    settings.diffuseMaxFastAccumulatedFrameNum =
        std::min(settings.diffuseMaxFastAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    settings.specularMaxFastAccumulatedFrameNum =
        std::min(settings.specularMaxFastAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    settings.historyFixFrameNum = std::min(settings.historyFixFrameNum, 3u);
    const uint32_t minimumFastHistory =
        std::min(settings.diffuseMaxFastAccumulatedFrameNum, settings.specularMaxFastAccumulatedFrameNum);
    settings.historyFixFrameNum =
        minimumFastHistory == 0 ? 0u : std::min(settings.historyFixFrameNum, minimumFastHistory - 1u);
    settings.historyFixBasePixelStride = std::max(settings.historyFixBasePixelStride, 1u);
    settings.historyFixAlternatePixelStride = std::max(settings.historyFixAlternatePixelStride, 1u);
    settings.historyFixEdgeStoppingNormalPower = std::max(settings.historyFixEdgeStoppingNormalPower, 0.0f);
    settings.fastHistoryClampingSigmaScale = std::clamp(settings.fastHistoryClampingSigmaScale, 1.0f, 3.0f);
    settings.diffusePrepassBlurRadius = std::max(settings.diffusePrepassBlurRadius, 0.0f);
    settings.specularPrepassBlurRadius = std::max(settings.specularPrepassBlurRadius, 0.0f);
    settings.minHitDistanceWeight = std::clamp(settings.minHitDistanceWeight, 0.0f, 0.2f);
    settings.spatialVarianceEstimationHistoryThreshold =
        std::min(settings.spatialVarianceEstimationHistoryThreshold, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    settings.diffusePhiLuminance = std::max(settings.diffusePhiLuminance, 0.0f);
    settings.specularPhiLuminance = std::max(settings.specularPhiLuminance, 0.0f);
    settings.lobeAngleFraction = std::clamp(settings.lobeAngleFraction, 0.0f, 1.0f);
    settings.roughnessFraction = std::clamp(settings.roughnessFraction, 0.0f, 1.0f);
    settings.specularVarianceBoost = std::max(settings.specularVarianceBoost, 0.0f);
    settings.specularLobeAngleSlack = std::max(settings.specularLobeAngleSlack, 0.0f);
    settings.atrousIterationNum = std::clamp(settings.atrousIterationNum, 2u, 8u);
    settings.diffuseMinLuminanceWeight = std::clamp(settings.diffuseMinLuminanceWeight, 0.0f, 1.0f);
    settings.specularMinLuminanceWeight = std::clamp(settings.specularMinLuminanceWeight, 0.0f, 1.0f);
    settings.depthThreshold = std::max(settings.depthThreshold, 0.0f);
    settings.confidenceDrivenRelaxationMultiplier = std::max(settings.confidenceDrivenRelaxationMultiplier, 0.0f);
    settings.confidenceDrivenLuminanceEdgeStoppingRelaxation =
        std::max(settings.confidenceDrivenLuminanceEdgeStoppingRelaxation, 0.0f);
    settings.confidenceDrivenNormalEdgeStoppingRelaxation =
        std::max(settings.confidenceDrivenNormalEdgeStoppingRelaxation, 0.0f);
    settings.luminanceEdgeStoppingRelaxation = std::max(settings.luminanceEdgeStoppingRelaxation, 0.0f);
    settings.normalEdgeStoppingRelaxation = std::max(settings.normalEdgeStoppingRelaxation, 0.0f);
    settings.roughnessEdgeStoppingRelaxation = std::max(settings.roughnessEdgeStoppingRelaxation, 0.0f);
    settings.minMaterialForDiffuse = std::max(settings.minMaterialForDiffuse, 0.0f);
    settings.minMaterialForSpecular = std::max(settings.minMaterialForSpecular, 0.0f);
}

void NrdModule::applyFrameRateTemporalSettings(float framesPerSecond) {
    const float fps = std::clamp(framesPerSecond, 5.0f, 240.0f);
    auto scaleFrameCount = [fps](uint32_t configuredFrames, uint32_t maximumFrames) {
        if (configuredFrames == 0) return 0u;
        const float accumulationSeconds = static_cast<float>(configuredFrames) / 60.0f;
        return std::clamp(nrd::GetMaxAccumulatedFrameNum(accumulationSeconds, fps), 1u, maximumFrames);
    };

    effectiveReblurSettings_ = reblurSettings_;
    effectiveReblurSettings_.maxAccumulatedFrameNum =
        scaleFrameCount(reblurSettings_.maxAccumulatedFrameNum, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
    effectiveReblurSettings_.maxFastAccumulatedFrameNum =
        std::min(reblurSettings_.maxFastAccumulatedFrameNum,
                 scaleFrameCount(reblurSettings_.maxFastAccumulatedFrameNum, nrd::REBLUR_MAX_HISTORY_FRAME_NUM));
    effectiveReblurSettings_.maxStabilizedFrameNum =
        nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard ?
            0u :
            scaleFrameCount(reblurSettings_.maxStabilizedFrameNum, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
    if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        effectiveReblurSettings_.maxAccumulatedFrameNum = std::min(reblurSettings_.maxAccumulatedFrameNum, 30u);
        effectiveReblurSettings_.maxFastAccumulatedFrameNum = std::min(reblurSettings_.maxFastAccumulatedFrameNum, 6u);
    }
    effectiveReblurSettings_.historyFixFrameNum =
        effectiveReblurSettings_.maxFastAccumulatedFrameNum == 0 ?
            0u :
            std::min(reblurSettings_.historyFixFrameNum, effectiveReblurSettings_.maxFastAccumulatedFrameNum - 1u);
    effectiveReblurSettings_.responsiveAccumulationSettings.minAccumulatedFrameNum =
        std::min(reblurSettings_.responsiveAccumulationSettings.minAccumulatedFrameNum,
                 effectiveReblurSettings_.historyFixFrameNum);
    if (isShEnabled_) { effectiveReblurSettings_.lobeAngleFraction *= 1.333f; }

    effectiveRelaxSettings_ = relaxSettings_;
    if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        effectiveRelaxSettings_.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
    }
    effectiveRelaxSettings_.diffuseMaxAccumulatedFrameNum =
        scaleFrameCount(relaxSettings_.diffuseMaxAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    effectiveRelaxSettings_.specularMaxAccumulatedFrameNum =
        scaleFrameCount(relaxSettings_.specularMaxAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
    effectiveRelaxSettings_.diffuseMaxFastAccumulatedFrameNum =
        std::min(relaxSettings_.diffuseMaxFastAccumulatedFrameNum,
                 scaleFrameCount(relaxSettings_.diffuseMaxFastAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
    effectiveRelaxSettings_.specularMaxFastAccumulatedFrameNum =
        std::min(relaxSettings_.specularMaxFastAccumulatedFrameNum,
                 scaleFrameCount(relaxSettings_.specularMaxFastAccumulatedFrameNum, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
    if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        effectiveRelaxSettings_.atrousIterationNum = 2;
        effectiveRelaxSettings_.diffuseMinLuminanceWeight = 0.0f;
        effectiveRelaxSettings_.specularMinLuminanceWeight = 0.0f;
    }
    const uint32_t effectiveMinimumFastHistory = std::min(effectiveRelaxSettings_.diffuseMaxFastAccumulatedFrameNum,
                                                          effectiveRelaxSettings_.specularMaxFastAccumulatedFrameNum);
    effectiveRelaxSettings_.historyFixFrameNum =
        effectiveMinimumFastHistory == 0 ?
            0u :
            std::min(relaxSettings_.historyFixFrameNum, effectiveMinimumFastHistory - 1u);

    if (indirectWrapper_) { indirectWrapper_->setReblurSettings(effectiveReblurSettings_); }
    if (directWrapper_ && isSeparateDirectDenoisingEnabled_) {
        directWrapper_->setRelaxSettings(effectiveRelaxSettings_);
    }
}

void NrdModule::updateFrameTime(float milliseconds, bool restart) {
    if (std::isfinite(milliseconds) && milliseconds > 0.0f) {
        const float seconds = std::clamp(milliseconds * 0.001f, 1.0f / 240.0f, 1.0f / 5.0f);
        const float fps = 1.0f / seconds;
        smoothedFramesPerSecond_ =
            restart ? fps : std::lerp(smoothedFramesPerSecond_, fps, 1.0f - std::exp(-seconds / 0.5f));
    }
    applyFrameRateTemporalSettings(smoothedFramesPerSecond_);
}

NrdModule::NrdModule() : reblurSettings_(makeDefaultReblurSettings()), relaxSettings_(makeDefaultRelaxSettings()) {}

NrdModule::~NrdModule() {
    indirectWrapper_ = nullptr;
    directWrapper_ = nullptr;
}

void NrdModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    directDiffuseRadianceImages_.resize(size);
    directSpecularRadianceImages_.resize(size);
    directHitDistanceImages_.resize(size);
    directLightGuideNormalRoughnessImages_.resize(size);
    directLightGuideMotionDepthImages_.resize(size);

    indirectDiffuseRadianceImages_.resize(size);
    indirectSpecularRadianceImages_.resize(size);
    indirectDiffuseDirectionImages_.resize(size);
    indirectSpecularDirectionImages_.resize(size);
    indirectDiffuseHitDistanceImages_.resize(size);
    indirectSpecularHitDistanceImages_.resize(size);
    indirectGuideNormalRoughnessImages_.resize(size);
    indirectGuideMotionImages_.resize(size);
    indirectGuideDepthImages_.resize(size);

    sharedDiffuseAlbedoMetallicImages_.resize(size);
    sharedSpecularAlbedoImages_.resize(size);

    composeClearRadianceImages_.resize(size);
    composeFirstHitRefractionImages_.resize(size);
    composeBaseEmissionImages_.resize(size);
    composeFogImages_.resize(size);
    denoisedRadianceImages_.resize(size);
}

bool NrdModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                       std::vector<VkFormat> &formats,
                                       uint32_t frameIndex) {
    if (images.size() != inputImageNum || formats.size() != inputImageNum) return false;

    auto framework = framework_.lock();

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            if (width_ == 0 || height_ == 0) { return false; }
            images[i] = createInputImage(framework, width_, height_, formats[i]);
        }
    }

    directDiffuseRadianceImages_[frameIndex] = images[DIRECT_DIFFUSE_RADIANCE_INPUT];
    directSpecularRadianceImages_[frameIndex] = images[DIRECT_SPECULAR_RADIANCE_INPUT];
    directHitDistanceImages_[frameIndex] = images[DIRECT_HIT_DISTANCE_INPUT];
    directLightGuideNormalRoughnessImages_[frameIndex] = images[DIRECT_LIGHT_GUIDE_NORMAL_ROUGHNESS_INPUT];
    directLightGuideMotionDepthImages_[frameIndex] = images[DIRECT_LIGHT_GUIDE_MOTION_DEPTH_INPUT];

    indirectDiffuseRadianceImages_[frameIndex] = images[INDIRECT_DIFFUSE_RADIANCE_INPUT];
    indirectSpecularRadianceImages_[frameIndex] = images[INDIRECT_SPECULAR_RADIANCE_INPUT];
    indirectDiffuseDirectionImages_[frameIndex] = images[INDIRECT_DIFFUSE_DIRECTION_INPUT];
    indirectSpecularDirectionImages_[frameIndex] = images[INDIRECT_SPECULAR_DIRECTION_INPUT];
    indirectDiffuseHitDistanceImages_[frameIndex] = images[INDIRECT_DIFFUSE_HIT_DISTANCE_INPUT];
    indirectSpecularHitDistanceImages_[frameIndex] = images[INDIRECT_SPECULAR_HIT_DISTANCE_INPUT];
    indirectGuideNormalRoughnessImages_[frameIndex] = images[INDIRECT_GUIDE_NORMAL_ROUGHNESS_INPUT];
    indirectGuideMotionImages_[frameIndex] = images[INDIRECT_GUIDE_MOTION_INPUT];
    indirectGuideDepthImages_[frameIndex] = images[INDIRECT_GUIDE_DEPTH_INPUT];

    sharedDiffuseAlbedoMetallicImages_[frameIndex] = images[SHARED_DIFFUSE_ALBEDO_METALLIC_INPUT];
    sharedSpecularAlbedoImages_[frameIndex] = images[SHARED_SPECULAR_ALBEDO_INPUT];

    composeClearRadianceImages_[frameIndex] = images[COMPOSE_CLEAR_RADIANCE_INPUT];
    composeFirstHitRefractionImages_[frameIndex] = images[COMPOSE_FIRST_HIT_REFRACTION_INPUT];
    composeBaseEmissionImages_[frameIndex] = images[COMPOSE_BASE_EMISSION_INPUT];
    composeFogImages_[frameIndex] = images[COMPOSE_FOG_INPUT];

    return true;
}

bool NrdModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    if (images.size() != outputImageNum) return false;
    if (images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    denoisedRadianceImages_[frameIndex] = images[0];

    return true;
}

void NrdModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    nrdMode_ = worldPipeline->shaderPack()->nrdMode();
    const bool halfRate = nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard;
    directWidth_ = width_;
    directHeight_ = height_;
    indirectWidth_ = width_;
    indirectHeight_ = height_;
    indirectWrapper_ = NrdWrapper::create(framework, static_cast<uint16_t>(indirectWidth_),
                                          static_cast<uint16_t>(indirectHeight_), isShEnabled_, false, true, halfRate);
    if (isSeparateDirectDenoisingEnabled_) {
        directWrapper_ = NrdWrapper::create(framework, static_cast<uint16_t>(directWidth_),
                                            static_cast<uint16_t>(directHeight_), false, true, false, false);
    } else {
        directWrapper_ = nullptr;
    }

    updateReblurSettings();
    if (isSeparateDirectDenoisingEnabled_) { updateRelaxSettings(); }
    applyFrameRateTemporalSettings(60.0f);
    nrdFrameIndex_ = 0;
    hasLastRenderTimestamp_ = false;
    smoothedFramesPerSecond_ = 60.0f;
    halfRateResolveParity_ = 0;
    halfRateResolveHasHistory_ = false;

    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (int i = 0; i < size; i++) {
        contexts_[i] = NrdModuleContext::create(
            framework->contexts()[i], worldPipeline ? worldPipeline->contexts()[i] : nullptr, shared_from_this());
    }
}

void NrdModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.nrd.attribute.enable_sh") {
            isShEnabled_ = parseBool(value);
        } else if (key == "render_pipeline.module.nrd.attribute.enable_separate_direct_denoising") {
            isSeparateDirectDenoisingEnabled_ = parseBool(value);
        } else if (key == "render_pipeline.module.nrd.attribute.max_accumulated_frame_num") {
            reblurSettings_.maxAccumulatedFrameNum = parseUint(value, reblurSettings_.maxAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.max_fast_accumulated_frame_num") {
            reblurSettings_.maxFastAccumulatedFrameNum = parseUint(value, reblurSettings_.maxFastAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.max_stabilized_frame_num") {
            reblurSettings_.maxStabilizedFrameNum = parseUint(value, reblurSettings_.maxStabilizedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.history_fix_frame_num") {
            reblurSettings_.historyFixFrameNum = parseUint(value, reblurSettings_.historyFixFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.history_fix_base_pixel_stride") {
            reblurSettings_.historyFixBasePixelStride = parseUint(value, reblurSettings_.historyFixBasePixelStride);
        } else if (key == "render_pipeline.module.nrd.attribute.history_fix_alternate_pixel_stride") {
            reblurSettings_.historyFixAlternatePixelStride =
                parseUint(value, reblurSettings_.historyFixAlternatePixelStride);
        } else if (key == "render_pipeline.module.nrd.attribute.fast_history_clamping_sigma_scale") {
            reblurSettings_.fastHistoryClampingSigmaScale =
                parseFloat(value, reblurSettings_.fastHistoryClampingSigmaScale);
        } else if (key == "render_pipeline.module.nrd.attribute.diffuse_prepass_blur_radius") {
            reblurSettings_.diffusePrepassBlurRadius = parseFloat(value, reblurSettings_.diffusePrepassBlurRadius);
        } else if (key == "render_pipeline.module.nrd.attribute.indirect_specular_prepass_blur_radius") {
            reblurSettings_.specularPrepassBlurRadius = parseFloat(value, reblurSettings_.specularPrepassBlurRadius);
        } else if (key == "render_pipeline.module.nrd.attribute.indirect_hit_distance_reconstruction_mode") {
            configuredReblurHitDistanceReconstructionMode_ =
                parseHitDistanceReconstructionMode(value, configuredReblurHitDistanceReconstructionMode_);
        } else if (key == "render_pipeline.module.nrd.attribute.min_hit_distance_weight") {
            reblurSettings_.minHitDistanceWeight = parseFloat(value, reblurSettings_.minHitDistanceWeight);
        } else if (key == "render_pipeline.module.nrd.attribute.min_blur_radius") {
            reblurSettings_.minBlurRadius = parseFloat(value, reblurSettings_.minBlurRadius);
        } else if (key == "render_pipeline.module.nrd.attribute.max_blur_radius") {
            reblurSettings_.maxBlurRadius = parseFloat(value, reblurSettings_.maxBlurRadius);
        } else if (key == "render_pipeline.module.nrd.attribute.lobe_angle_fraction") {
            reblurSettings_.lobeAngleFraction = parseFloat(value, reblurSettings_.lobeAngleFraction);
        } else if (key == "render_pipeline.module.nrd.attribute.roughness_fraction") {
            reblurSettings_.roughnessFraction = parseFloat(value, reblurSettings_.roughnessFraction);
        } else if (key == "render_pipeline.module.nrd.attribute.plane_distance_sensitivity") {
            reblurSettings_.planeDistanceSensitivity = parseFloat(value, reblurSettings_.planeDistanceSensitivity);
        } else if (key == "render_pipeline.module.nrd.attribute.specular_probability_threshold_low") {
            reblurSettings_.specularProbabilityThresholdsForMvModification[0] =
                parseFloat(value, reblurSettings_.specularProbabilityThresholdsForMvModification[0]);
        } else if (key == "render_pipeline.module.nrd.attribute.specular_probability_threshold_high") {
            reblurSettings_.specularProbabilityThresholdsForMvModification[1] =
                parseFloat(value, reblurSettings_.specularProbabilityThresholdsForMvModification[1]);
        } else if (key == "render_pipeline.module.nrd.attribute.firefly_suppressor_min_relative_scale") {
            reblurSettings_.fireflySuppressorMinRelativeScale =
                parseFloat(value, reblurSettings_.fireflySuppressorMinRelativeScale);
        } else if (key == "render_pipeline.module.nrd.attribute.min_material_for_diffuse") {
            reblurSettings_.minMaterialForDiffuse = parseFloat(value, reblurSettings_.minMaterialForDiffuse);
        } else if (key == "render_pipeline.module.nrd.attribute.min_material_for_specular") {
            reblurSettings_.minMaterialForSpecular = parseFloat(value, reblurSettings_.minMaterialForSpecular);
        } else if (key == "render_pipeline.module.nrd.attribute.enable_anti_firefly") {
            reblurSettings_.enableAntiFirefly = parseBool(value);
        } else if (key == "render_pipeline.module.nrd.attribute.use_prepass_only_for_specular_motion_estimation") {
            reblurSettings_.usePrepassOnlyForSpecularMotionEstimation = parseBool(value);
        } else if (key == "render_pipeline.module.nrd.attribute.return_history_length_instead_of_occlusion") {
            reblurSettings_.returnHistoryLengthInsteadOfOcclusion = parseBool(value);
        } else if (key == "render_pipeline.module.nrd.attribute.antilag_luminance_sigma_scale") {
            reblurSettings_.antilagSettings.luminanceSigmaScale =
                parseFloat(value, reblurSettings_.antilagSettings.luminanceSigmaScale);
        } else if (key == "render_pipeline.module.nrd.attribute.antilag_luminance_sensitivity") {
            reblurSettings_.antilagSettings.luminanceSensitivity =
                parseFloat(value, reblurSettings_.antilagSettings.luminanceSensitivity);
        } else if (key == "render_pipeline.module.nrd.attribute.responsive_roughness_threshold") {
            reblurSettings_.responsiveAccumulationSettings.roughnessThreshold =
                parseFloat(value, reblurSettings_.responsiveAccumulationSettings.roughnessThreshold);
        } else if (key == "render_pipeline.module.nrd.attribute.responsive_min_accumulated_frame_num") {
            reblurSettings_.responsiveAccumulationSettings.minAccumulatedFrameNum =
                parseUint(value, reblurSettings_.responsiveAccumulationSettings.minAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_diffuse_max_accumulated_frame_num") {
            relaxSettings_.diffuseMaxAccumulatedFrameNum =
                parseUint(value, relaxSettings_.diffuseMaxAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_specular_max_accumulated_frame_num") {
            relaxSettings_.specularMaxAccumulatedFrameNum =
                parseUint(value, relaxSettings_.specularMaxAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_diffuse_max_fast_accumulated_frame_num") {
            relaxSettings_.diffuseMaxFastAccumulatedFrameNum =
                parseUint(value, relaxSettings_.diffuseMaxFastAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_specular_max_fast_accumulated_frame_num") {
            relaxSettings_.specularMaxFastAccumulatedFrameNum =
                parseUint(value, relaxSettings_.specularMaxFastAccumulatedFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_history_fix_frame_num") {
            relaxSettings_.historyFixFrameNum = parseUint(value, relaxSettings_.historyFixFrameNum);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_fast_history_clamping_sigma_scale") {
            relaxSettings_.fastHistoryClampingSigmaScale =
                parseFloat(value, relaxSettings_.fastHistoryClampingSigmaScale);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_diffuse_prepass_blur_radius") {
            relaxSettings_.diffusePrepassBlurRadius = parseFloat(value, relaxSettings_.diffusePrepassBlurRadius);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_specular_prepass_blur_radius") {
            relaxSettings_.specularPrepassBlurRadius = parseFloat(value, relaxSettings_.specularPrepassBlurRadius);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_min_hit_distance_weight") {
            relaxSettings_.minHitDistanceWeight = parseFloat(value, relaxSettings_.minHitDistanceWeight);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_antilag_acceleration_amount") {
            relaxSettings_.antilagSettings.accelerationAmount =
                parseFloat(value, relaxSettings_.antilagSettings.accelerationAmount);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_antilag_reset_amount") {
            relaxSettings_.antilagSettings.resetAmount = parseFloat(value, relaxSettings_.antilagSettings.resetAmount);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_hit_distance_reconstruction_mode") {
            relaxSettings_.hitDistanceReconstructionMode =
                parseHitDistanceReconstructionMode(value, relaxSettings_.hitDistanceReconstructionMode);
        } else if (key == "render_pipeline.module.nrd.attribute.direct_enable_anti_firefly") {
            relaxSettings_.enableAntiFirefly = parseBool(value);
        }
    }

    updateReblurSettings();
    updateRelaxSettings();
}

std::vector<std::shared_ptr<WorldModuleContext>> &NrdModule::contexts() {
    return contexts_;
}

void NrdModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                            std::shared_ptr<vk::DeviceLocalImage> image,
                            int index) {}

void NrdModule::preClose() {
    indirectWrapper_ = nullptr;
    directWrapper_ = nullptr;
}

void NrdModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    composeDescriptorTables_.resize(size);
    prepareDescriptorTables_.resize(size);
    lightingResponseTables_.assign(size, {});
    halfRateResolveTables_.clear();
    halfRateSecondaryResolveTables_.clear();
    if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        halfRateResolveTables_.resize(size);
        halfRateSecondaryResolveTables_.resize(size);
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
            for (auto &table : lightingResponseTables_[i]) {
                vk::DescriptorTableBuilder builder;
                auto &bindings = builder.beginDescriptorLayoutSet().beginStaticDescriptorLayoutSetBinding();
                for (uint32_t binding : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 17u, 18u})
                    bindings.defineDescriptorLayoutSetBinding({binding, binding < 6 || binding == 8 || binding == 9 || binding >= 12 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
                table = bindings.endDescriptorLayoutSetBinding().endDescriptorLayoutSet()
                    .definePushConstant({VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(NrdLightingPushConstants)}).build(framework->device());
            }
            for (auto *tables : {&halfRateResolveTables_[i], &halfRateSecondaryResolveTables_[i]}) {
                for (auto &table : *tables) {
                    vk::DescriptorTableBuilder builder;
                    auto &bindings = builder.beginDescriptorLayoutSet().beginStaticDescriptorLayoutSetBinding();
                    for (uint32_t binding : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 10u, 11u, 12u})
                        bindings.defineDescriptorLayoutSetBinding(
                            {.binding = binding,
                             .descriptorType = binding < 6 || (binding >= 9 && binding <= 12) ?
                                                   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                                                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = 1,
                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
                    table = bindings.endDescriptorLayoutSetBinding()
                                .endDescriptorLayoutSet()
                                .definePushConstant({VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(NrdResolvePushConstants)})
                                .build(framework->device());
                }
            }
        }
        {
            auto builder = vk::DescriptorTableBuilder{};
            auto &set0BindingBuilder = builder.beginDescriptorLayoutSet().beginDescriptorLayoutSetBinding();

            for (uint32_t binding : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u}) {
                set0BindingBuilder.defineDescriptorLayoutSetBinding({
                    .binding = binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                });
            }

            auto &set1BindingBuilder = set0BindingBuilder.endDescriptorLayoutSetBinding()
                                           .endDescriptorLayoutSet()
                                           .beginDescriptorLayoutSet()
                                           .beginDescriptorLayoutSetBinding();

            for (uint32_t binding :
                 {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u}) {
                set1BindingBuilder.defineDescriptorLayoutSetBinding({
                    .binding = binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                });
            }

            auto &set2BindingBuilder = set1BindingBuilder.endDescriptorLayoutSetBinding()
                                           .endDescriptorLayoutSet()
                                           .beginDescriptorLayoutSet()
                                           .beginDescriptorLayoutSetBinding();
            set2BindingBuilder.defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });

            set2BindingBuilder.defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
            set2BindingBuilder.defineDescriptorLayoutSetBinding({
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
            prepareDescriptorTables_[i] = set2BindingBuilder.endDescriptorLayoutSetBinding()
                                              .endDescriptorLayoutSet()
                                              .definePushConstant({
                                                  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                  .offset = 0,
                                                  .size = sizeof(NrdModePushConstants),
                                              })
                                              .build(framework->device());
        }

        {
            auto builder = vk::DescriptorTableBuilder{};
            auto &set0BindingBuilder = builder.beginDescriptorLayoutSet().beginDescriptorLayoutSetBinding();

            for (uint32_t binding : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u}) {
                set0BindingBuilder.defineDescriptorLayoutSetBinding({
                    .binding = binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                });
            }

            auto &set1BindingBuilder = set0BindingBuilder.endDescriptorLayoutSetBinding()
                                           .endDescriptorLayoutSet()
                                           .beginDescriptorLayoutSet()
                                           .beginDescriptorLayoutSetBinding();

            for (uint32_t binding = 0; binding < 13; ++binding) {
                set1BindingBuilder.defineDescriptorLayoutSetBinding({
                    .binding = binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                });
            }

            auto &set2BindingBuilder = set1BindingBuilder.endDescriptorLayoutSetBinding()
                                           .endDescriptorLayoutSet()
                                           .beginDescriptorLayoutSet()
                                           .beginDescriptorLayoutSetBinding();

            set2BindingBuilder.defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
            set2BindingBuilder.defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });

            composeDescriptorTables_[i] = set2BindingBuilder.endDescriptorLayoutSetBinding()
                                              .endDescriptorLayoutSet()
                                              .definePushConstant({
                                                  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                  .offset = 0,
                                                  .size = sizeof(NrdModePushConstants),
                                              })
                                              .build(framework->device());
        }
    }
}

void NrdModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    lightingResponseHistory_ = {};
    rawLightingHistory_ = {};
    lightingResponseImages_.assign(size, nullptr);
    halfRateResolveHistory_ = {};
    halfRateSecondaryHistory_ = {};
    if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        const size_t tileBytes = size_t((width_ + 31) / 32) * ((height_ + 31) / 32) * 8 * sizeof(glm::vec4);
        for (auto &history : lightingResponseHistory_)
            history = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), false, tileBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const size_t pixels = size_t(width_) * height_;
        for (auto &history : rawLightingHistory_)
            history = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), false,
                2 * pixels * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const size_t bytes = 2 * pixels * sizeof(glm::uvec2);
        for (auto &history : halfRateResolveHistory_)
            history = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), false, bytes,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (auto &history : halfRateSecondaryHistory_)
            history = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), false, bytes,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    nrdIndirectDiffuseRadianceImages_.resize(size);
    nrdIndirectSpecularRadianceImages_.resize(size);
    composedLinearImages_.resize(size);
    halfRateIndirectImages_.assign(size, nullptr);
    halfRateSecondaryImages_.assign(size, nullptr);
    nrdIndirectDiffuseSh0Images_.resize(size);
    nrdIndirectDiffuseSh1Images_.resize(size);
    nrdIndirectSpecularSh0Images_.resize(size);
    nrdIndirectSpecularSh1Images_.resize(size);
    nrdIndirectMotionVectorImages_.resize(size);
    nrdIndirectNormalRoughnessImages_.resize(size);
    nrdIndirectLinearDepthImages_.resize(size);
    nrdIndirectBaseColorMetalnessImages_.resize(size);
    nrdDirectMotionVectorImages_.resize(size);
    nrdDirectNormalRoughnessImages_.resize(size);
    nrdDirectLinearDepthImages_.resize(size);
    nrdDirectDiffuseRadianceImages_.resize(size);
    nrdDirectSpecularRadianceImages_.resize(size);
    nrdDirectDisocclusionThresholdMixImages_.resize(size);
    nrdIndirectDisocclusionThresholdMixImages_.resize(size);
    nrdDirectConfidenceImages_.resize(size);
    nrdIndirectConfidenceImages_.resize(size);
    halfRateDiffuseGuides_.assign(size, {});
    denoisedIndirectDiffuseRadianceImages_.resize(size);
    denoisedIndirectSpecularRadianceImages_.resize(size);
    denoisedIndirectDiffuseSh0Images_.resize(size);
    denoisedIndirectDiffuseSh1Images_.resize(size);
    denoisedIndirectSpecularSh0Images_.resize(size);
    denoisedIndirectSpecularSh1Images_.resize(size);
    denoisedDirectDiffuseRadianceImages_.resize(size);
    denoisedDirectSpecularRadianceImages_.resize(size);
    indirectTexturePools_.resize(size);
    directTexturePools_.resize(size);

    auto createRgba16Image = [&](uint32_t imageWidth, uint32_t imageHeight) {
        return vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, imageWidth, imageHeight, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    };
    auto createImage = [&](VkFormat format, uint32_t imageWidth, uint32_t imageHeight) {
        return vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, imageWidth, imageHeight, 1, format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    };

    for (uint32_t i = 0; i < size; i++) {
        indirectTexturePools_[i] =
            std::make_shared<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>();
        directTexturePools_[i] =
            std::make_shared<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>();

        prepareDescriptorTables_[i]->bindImage(indirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        prepareDescriptorTables_[i]->bindImage(indirectDiffuseDirectionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        prepareDescriptorTables_[i]->bindImage(indirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        prepareDescriptorTables_[i]->bindImage(indirectSpecularDirectionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);
        prepareDescriptorTables_[i]->bindImage(sharedDiffuseAlbedoMetallicImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        prepareDescriptorTables_[i]->bindImage(sharedSpecularAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        prepareDescriptorTables_[i]->bindImage(indirectGuideMotionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 6);
        prepareDescriptorTables_[i]->bindImage(indirectGuideNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 7);
        prepareDescriptorTables_[i]->bindImage(indirectGuideDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 8);
        prepareDescriptorTables_[i]->bindImage(composeClearRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 9);
        prepareDescriptorTables_[i]->bindImage(indirectDiffuseHitDistanceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 10);
        prepareDescriptorTables_[i]->bindImage(indirectSpecularHitDistanceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 11);
        prepareDescriptorTables_[i]->bindImage(directDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 12);
        prepareDescriptorTables_[i]->bindImage(directSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 13);
        prepareDescriptorTables_[i]->bindImage(directHitDistanceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 14);
        prepareDescriptorTables_[i]->bindImage(composeBaseEmissionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 15);
        prepareDescriptorTables_[i]->bindImage(directLightGuideNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0,
                                               16);
        prepareDescriptorTables_[i]->bindImage(directLightGuideMotionDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 17);

        nrdIndirectMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, indirectWidth_, indirectHeight_, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        prepareDescriptorTables_[i]->bindImage(nrdIndirectMotionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 6);

        nrdIndirectNormalRoughnessImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, indirectWidth_, indirectHeight_, 1,
            NrdWrapper::getNormalRoughnessFormat(),
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        prepareDescriptorTables_[i]->bindImage(nrdIndirectNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 7);

        nrdIndirectLinearDepthImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, indirectWidth_, indirectHeight_, 1, VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        prepareDescriptorTables_[i]->bindImage(nrdIndirectLinearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 8);
        nrdIndirectBaseColorMetalnessImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, indirectWidth_, indirectHeight_, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        prepareDescriptorTables_[i]->bindImage(nrdIndirectBaseColorMetalnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 18);

        nrdDirectMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, directWidth_, directHeight_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        nrdDirectNormalRoughnessImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, directWidth_, directHeight_, 1,
            NrdWrapper::getNormalRoughnessFormat(),
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        nrdDirectLinearDepthImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, directWidth_, directHeight_, 1, VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        prepareDescriptorTables_[i]->bindImage(nrdDirectMotionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 15);
        prepareDescriptorTables_[i]->bindImage(nrdDirectNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 16);
        prepareDescriptorTables_[i]->bindImage(nrdDirectLinearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 17);
        composeDescriptorTables_[i]->bindImage(nrdDirectLinearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 9);
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_MV) = nrdIndirectMotionVectorImages_[i];
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_NORMAL_ROUGHNESS) =
            nrdIndirectNormalRoughnessImages_[i];
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_VIEWZ) = nrdIndirectLinearDepthImages_[i];
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_BASECOLOR_METALNESS) =
            nrdIndirectBaseColorMetalnessImages_[i];
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_MV) = nrdDirectMotionVectorImages_[i];
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_NORMAL_ROUGHNESS) = nrdDirectNormalRoughnessImages_[i];
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_VIEWZ) = nrdDirectLinearDepthImages_[i];

        if (isShEnabled_) {
            nrdIndirectDiffuseSh0Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            nrdIndirectDiffuseSh1Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            nrdIndirectSpecularSh0Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            nrdIndirectSpecularSh1Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);

            prepareDescriptorTables_[i]->bindImage(nrdIndirectDiffuseSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 0);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectSpecularSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 1);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectDiffuseSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 2);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectDiffuseSh1Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 3);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectSpecularSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 4);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectSpecularSh1Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 5);

            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DIFF_SH0) = nrdIndirectDiffuseSh0Images_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DIFF_SH1) = nrdIndirectDiffuseSh1Images_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_SPEC_SH0) = nrdIndirectSpecularSh0Images_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_SPEC_SH1) = nrdIndirectSpecularSh1Images_[i];

            denoisedIndirectDiffuseSh0Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            denoisedIndirectDiffuseSh1Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            denoisedIndirectSpecularSh0Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            denoisedIndirectSpecularSh1Images_[i] = createRgba16Image(indirectWidth_, indirectHeight_);

            composeDescriptorTables_[i]->bindImage(denoisedIndirectDiffuseSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 0);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectSpecularSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                                   1);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectDiffuseSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 2);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectDiffuseSh1Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 3);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectSpecularSh0Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                                   4);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectSpecularSh1Images_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                                   5);

            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_DIFF_SH0) =
                denoisedIndirectDiffuseSh0Images_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_DIFF_SH1) =
                denoisedIndirectDiffuseSh1Images_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_SPEC_SH0) =
                denoisedIndirectSpecularSh0Images_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_SPEC_SH1) =
                denoisedIndirectSpecularSh1Images_[i];
        } else {
            nrdIndirectDiffuseRadianceImages_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            nrdIndirectSpecularRadianceImages_[i] = createRgba16Image(indirectWidth_, indirectHeight_);

            prepareDescriptorTables_[i]->bindImage(nrdIndirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 0);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                                   1);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 2);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 3);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                                   4);
            prepareDescriptorTables_[i]->bindImage(nrdIndirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                                   5);

            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST) =
                nrdIndirectDiffuseRadianceImages_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST) =
                nrdIndirectSpecularRadianceImages_[i];

            denoisedIndirectDiffuseRadianceImages_[i] = createRgba16Image(indirectWidth_, indirectHeight_);
            denoisedIndirectSpecularRadianceImages_[i] = createRgba16Image(indirectWidth_, indirectHeight_);

            composeDescriptorTables_[i]->bindImage(denoisedIndirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                                   1, 0);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                                   1, 1);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                                   1, 2);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                                   1, 3);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                                   1, 4);
            composeDescriptorTables_[i]->bindImage(denoisedIndirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                                   1, 5);

            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST) =
                denoisedIndirectDiffuseRadianceImages_[i];
            indirectTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST) =
                denoisedIndirectSpecularRadianceImages_[i];
        }

        nrdDirectDiffuseRadianceImages_[i] = createRgba16Image(directWidth_, directHeight_);
        nrdDirectSpecularRadianceImages_[i] = createRgba16Image(directWidth_, directHeight_);
        prepareDescriptorTables_[i]->bindImage(nrdDirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 9);
        prepareDescriptorTables_[i]->bindImage(nrdDirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 10);
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST) =
            nrdDirectDiffuseRadianceImages_[i];
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST) =
            nrdDirectSpecularRadianceImages_[i];

        denoisedDirectDiffuseRadianceImages_[i] = createRgba16Image(directWidth_, directHeight_);
        denoisedDirectSpecularRadianceImages_[i] = createRgba16Image(directWidth_, directHeight_);
        composeDescriptorTables_[i]->bindImage(denoisedDirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 7);
        composeDescriptorTables_[i]->bindImage(denoisedDirectSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 8);
        directTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST) =
            denoisedDirectDiffuseRadianceImages_[i];
        directTexturePools_[i]->at((size_t)nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST) =
            denoisedDirectSpecularRadianceImages_[i];

        nrdDirectDisocclusionThresholdMixImages_[i] = createImage(VK_FORMAT_R8_UNORM, directWidth_, directHeight_);
        nrdIndirectDisocclusionThresholdMixImages_[i] =
            createImage(VK_FORMAT_R8_UNORM, indirectWidth_, indirectHeight_);
        prepareDescriptorTables_[i]->bindImage(nrdDirectDisocclusionThresholdMixImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1,
                                               11);
        prepareDescriptorTables_[i]->bindImage(nrdIndirectDisocclusionThresholdMixImages_[i], VK_IMAGE_LAYOUT_GENERAL,
                                               1, 12);
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX) =
            nrdDirectDisocclusionThresholdMixImages_[i];
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX) =
            nrdIndirectDisocclusionThresholdMixImages_[i];

        nrdDirectConfidenceImages_[i] = createImage(VK_FORMAT_R8_UNORM, directWidth_, directHeight_);
        nrdIndirectConfidenceImages_[i] = createImage(VK_FORMAT_R8_UNORM, indirectWidth_, indirectHeight_);
        prepareDescriptorTables_[i]->bindImage(nrdDirectConfidenceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 13);
        prepareDescriptorTables_[i]->bindImage(nrdIndirectConfidenceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 14);
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DIFF_CONFIDENCE) = nrdDirectConfidenceImages_[i];
        directTexturePools_[i]->at((size_t)nrd::ResourceType::IN_SPEC_CONFIDENCE) = nrdDirectConfidenceImages_[i];
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_DIFF_CONFIDENCE) = nrdIndirectConfidenceImages_[i];
        indirectTexturePools_[i]->at((size_t)nrd::ResourceType::IN_SPEC_CONFIDENCE) = nrdIndirectConfidenceImages_[i];

        if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
            lightingResponseImages_[i] = createImage(VK_FORMAT_R8G8_UNORM, (width_ + 31) / 32, ((height_ + 31) / 32) * 9);
            for (uint32_t parity = 0; parity < 2; ++parity) {
                auto &table = lightingResponseTables_[i][parity];
                table->bindImage(indirectDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
                table->bindImage(directDiffuseRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
                table->bindImage(sharedDiffuseAlbedoMetallicImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
                table->bindImage(indirectGuideDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);
                table->bindImage(indirectGuideMotionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 4);
                table->bindImage(lightingResponseImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
                table->bindBuffer(lightingResponseHistory_[1 - parity], 0, 6);
                table->bindBuffer(lightingResponseHistory_[parity], 0, 7);
                table->bindImage(directSpecularRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 8);
                table->bindImage(composeBaseEmissionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 9);
                table->bindBuffer(rawLightingHistory_[1 - parity], 0, 10);
                table->bindBuffer(rawLightingHistory_[parity], 0, 11);
                table->bindImage(sharedSpecularAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 12);
                table->bindImage(nrdDirectConfidenceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 13);
                table->bindImage(directHitDistanceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 14);
                table->bindImage(directLightGuideMotionDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 17);
                table->bindImage(indirectGuideNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 18);
            }
            prepareDescriptorTables_[i]->bindImage(lightingResponseImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 18);
        } else {
            prepareDescriptorTables_[i]->bindImage(nrdIndirectConfidenceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 18);
        }
        auto &diffuseGuide = halfRateDiffuseGuides_[i];
        const bool halfRate = nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard;
        diffuseGuide.motion =
            halfRate ? createImage(VK_FORMAT_R16G16B16A16_SFLOAT, width_, height_) : nrdIndirectMotionVectorImages_[i];
        diffuseGuide.normalRoughness = halfRate ? createImage(NrdWrapper::getNormalRoughnessFormat(), width_, height_) :
                                                  nrdIndirectNormalRoughnessImages_[i];
        diffuseGuide.depth =
            halfRate ? createImage(VK_FORMAT_R16_SFLOAT, width_, height_) : nrdIndirectLinearDepthImages_[i];
        prepareDescriptorTables_[i]->bindImage(diffuseGuide.motion, VK_IMAGE_LAYOUT_GENERAL, 1, 19);
        prepareDescriptorTables_[i]->bindImage(diffuseGuide.normalRoughness, VK_IMAGE_LAYOUT_GENERAL, 1, 20);
        prepareDescriptorTables_[i]->bindImage(diffuseGuide.depth, VK_IMAGE_LAYOUT_GENERAL, 1, 21);
        composeDescriptorTables_[i]->bindImage(diffuseGuide.motion, VK_IMAGE_LAYOUT_GENERAL, 1, 11);

        composeDescriptorTables_[i]->bindImage(sharedDiffuseAlbedoMetallicImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        composeDescriptorTables_[i]->bindImage(nrdIndirectNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 9);
        composeDescriptorTables_[i]->bindImage(sharedSpecularAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        composeDescriptorTables_[i]->bindImage(indirectGuideNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        composeDescriptorTables_[i]->bindImage(indirectGuideDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);
        composeDescriptorTables_[i]->bindImage(composeClearRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        const auto &firstHitRefractionImage = composeFirstHitRefractionImages_[i];
        if (firstHitRefractionImage == nullptr) {
            throw std::runtime_error("NRD refraction composition input is not connected");
        }
        composeDescriptorTables_[i]->bindImage(firstHitRefractionImage, VK_IMAGE_LAYOUT_GENERAL, 0, 10);
        if (i == 0) {
            std::fprintf(stderr, "[RADIANCE-E16] nrdComposeFirstHitRefraction=bound descriptor=0:10 extent=%ux%u\n",
                         firstHitRefractionImage->width(), firstHitRefractionImage->height());
        }
        composeDescriptorTables_[i]->bindImage(composeBaseEmissionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        composeDescriptorTables_[i]->bindImage(directLightGuideNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0,
                                               6);
        composeDescriptorTables_[i]->bindImage(directLightGuideMotionDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 7);
        composeDescriptorTables_[i]->bindImage(composeFogImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 8);

        composedLinearImages_[i] = nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard ?
                                       createImage(VK_FORMAT_R16G16B16A16_SFLOAT, width_, height_) :
                                       denoisedRadianceImages_[i];
        composeDescriptorTables_[i]->bindImage(composedLinearImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 6);

        halfRateIndirectImages_[i] = nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard ?
                                         createImage(VK_FORMAT_R16G16B16A16_SFLOAT, width_, height_) :
                                         composedLinearImages_[i];
        composeDescriptorTables_[i]->bindImage(halfRateIndirectImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 10);
        halfRateSecondaryImages_[i] =
            halfRate ? createImage(VK_FORMAT_R16G16B16A16_SFLOAT, width_, height_) : composedLinearImages_[i];
        composeDescriptorTables_[i]->bindImage(halfRateSecondaryImages_[i], VK_IMAGE_LAYOUT_GENERAL, 1, 12);
        if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
            const std::array images{composedLinearImages_[i],         nrdIndirectNormalRoughnessImages_[i],
                                    nrdIndirectLinearDepthImages_[i], nrdIndirectMotionVectorImages_[i],
                                    sharedSpecularAlbedoImages_[i],   denoisedRadianceImages_[i]};
            for (uint32_t parity = 0; parity < 2; ++parity) {
                auto &table = halfRateResolveTables_[i][parity];
                for (uint32_t binding = 0; binding < images.size(); ++binding)
                    table->bindImage(images[binding], VK_IMAGE_LAYOUT_GENERAL, 0, binding);
                table->bindBuffer(halfRateResolveHistory_[1 - parity], 0, 7);
                table->bindBuffer(halfRateResolveHistory_[parity], 0, 8);
                table->bindImage(halfRateIndirectImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 9);
                table->bindImage(sharedDiffuseAlbedoMetallicImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 10);
                table->bindImage(diffuseGuide.motion, VK_IMAGE_LAYOUT_GENERAL, 0, 11);
                table->bindImage(nrdIndirectConfidenceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 12);

                auto &secondaryTable = halfRateSecondaryResolveTables_[i][parity];
                const std::array secondaryImages{
                    composedLinearImages_[i], diffuseGuide.normalRoughness,   diffuseGuide.depth,
                    diffuseGuide.motion,      sharedSpecularAlbedoImages_[i], denoisedRadianceImages_[i]};
                for (uint32_t binding = 0; binding < secondaryImages.size(); ++binding)
                    secondaryTable->bindImage(secondaryImages[binding], VK_IMAGE_LAYOUT_GENERAL, 0, binding);
                secondaryTable->bindBuffer(halfRateSecondaryHistory_[1 - parity], 0, 7);
                secondaryTable->bindBuffer(halfRateSecondaryHistory_[parity], 0, 8);
                secondaryTable->bindImage(halfRateSecondaryImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 9);
                secondaryTable->bindImage(sharedDiffuseAlbedoMetallicImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 10);
                secondaryTable->bindImage(diffuseGuide.motion, VK_IMAGE_LAYOUT_GENERAL, 0, 11);
                secondaryTable->bindImage(nrdIndirectConfidenceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 12);
            }
        }
    }
}

void NrdModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    lightingResponsePipeline_.reset();
    halfRateResolvePipelines_ = {};
    if (nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        lightingResponsePipeline_ = vk::ComputePipelineBuilder{}
            .defineShader(vk::Shader::create(device, (shaderPath / "world/nrd/lighting_response_comp.spv").string()))
            .definePipelineLayout(lightingResponseTables_[0][0]).build(device);
        auto shader = vk::Shader::create(device, (shaderPath / "world/nrd/final_resolve_comp.spv").string());
        for (uint32_t variant = 0; variant < halfRateResolvePipelines_.size(); ++variant) {
            halfRateResolvePipelines_[variant] = vk::ComputePipelineBuilder{}
                                                     .defineShader(shader)
                                                     .definePipelineLayout(halfRateResolveTables_[0][0])
                                                     .defineSpecializationConstants({variant, 16, 16})
                                                     .build(device);
        }
    }

    auto prepareShader = vk::Shader::create(device, (shaderPath / "world/nrd/prepare_comp.spv").string());
    preparePipeline_ =
        vk::ComputePipelineBuilder{}
            .defineShader(prepareShader)
            .definePipelineLayout(prepareDescriptorTables_[0])
            .defineSpecializationConstants({static_cast<uint32_t>(nrdMode_), static_cast<uint32_t>(isShEnabled_),
                                            static_cast<uint32_t>(isSeparateDirectDenoisingEnabled_)})
            .build(device);

    const auto composePath = nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard ?
                                 "world/nrd/compose_half_rate_comp.spv" :
                                 "world/nrd/compose_comp.spv";
    auto composeShader = vk::Shader::create(device, (shaderPath / composePath).string());
    composePipeline_ = vk::ComputePipelineBuilder{}
                           .defineShader(composeShader)
                           .definePipelineLayout(composeDescriptorTables_[0])
                           .build(device);
}

void NrdModuleContext::addComputeImageBarrier(std::vector<vk::CommandBuffer::ImageMemoryBarrier> &barriers,
                                              const std::shared_ptr<vk::DeviceLocalImage> &image,
                                              VkImageLayout targetLayout,
                                              uint32_t queueIndex) {
    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    VkAccessFlags2 srcAccess = 0;
    if (image->imageLayout() == VK_IMAGE_LAYOUT_GENERAL) {
        srcStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        srcAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    } else if (image->imageLayout() == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        srcAccess = VK_ACCESS_2_SHADER_READ_BIT;
    }
    barriers.push_back({
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout = image->imageLayout(),
        .newLayout = targetLayout,
        .srcQueueFamilyIndex = queueIndex,
        .dstQueueFamilyIndex = queueIndex,
        .image = image,
        .subresourceRange = vk::wholeColorSubresourceRange,
    });
    image->imageLayout() = targetLayout;
}

NrdModuleContext::NrdModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                   std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                   std::shared_ptr<NrdModule> nrdModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      nrdModule(nrdModule),
      directDiffuseRadianceImage(nrdModule->directDiffuseRadianceImages_[frameworkContext->frameIndex]),
      directSpecularRadianceImage(nrdModule->directSpecularRadianceImages_[frameworkContext->frameIndex]),
      directHitDistanceImage(nrdModule->directHitDistanceImages_[frameworkContext->frameIndex]),
      directLightGuideNormalRoughnessImage(
          nrdModule->directLightGuideNormalRoughnessImages_[frameworkContext->frameIndex]),
      directLightGuideMotionDepthImage(nrdModule->directLightGuideMotionDepthImages_[frameworkContext->frameIndex]),
      indirectDiffuseRadianceImage(nrdModule->indirectDiffuseRadianceImages_[frameworkContext->frameIndex]),
      indirectSpecularRadianceImage(nrdModule->indirectSpecularRadianceImages_[frameworkContext->frameIndex]),
      indirectDiffuseDirectionImage(nrdModule->indirectDiffuseDirectionImages_[frameworkContext->frameIndex]),
      indirectSpecularDirectionImage(nrdModule->indirectSpecularDirectionImages_[frameworkContext->frameIndex]),
      indirectDiffuseHitDistanceImage(nrdModule->indirectDiffuseHitDistanceImages_[frameworkContext->frameIndex]),
      indirectSpecularHitDistanceImage(nrdModule->indirectSpecularHitDistanceImages_[frameworkContext->frameIndex]),
      indirectGuideNormalRoughnessImage(nrdModule->indirectGuideNormalRoughnessImages_[frameworkContext->frameIndex]),
      indirectGuideMotionImage(nrdModule->indirectGuideMotionImages_[frameworkContext->frameIndex]),
      indirectGuideDepthImage(nrdModule->indirectGuideDepthImages_[frameworkContext->frameIndex]),
      sharedDiffuseAlbedoMetallicImage(nrdModule->sharedDiffuseAlbedoMetallicImages_[frameworkContext->frameIndex]),
      sharedSpecularAlbedoImage(nrdModule->sharedSpecularAlbedoImages_[frameworkContext->frameIndex]),
      composeClearRadianceImage(nrdModule->composeClearRadianceImages_[frameworkContext->frameIndex]),
      composeFirstHitRefractionImage(nrdModule->composeFirstHitRefractionImages_[frameworkContext->frameIndex]),
      composeBaseEmissionImage(nrdModule->composeBaseEmissionImages_[frameworkContext->frameIndex]),
      composeFogImage(nrdModule->composeFogImages_[frameworkContext->frameIndex]),
      nrdIndirectDiffuseRadianceImage(nrdModule->nrdIndirectDiffuseRadianceImages_[frameworkContext->frameIndex]),
      nrdIndirectSpecularRadianceImage(nrdModule->nrdIndirectSpecularRadianceImages_[frameworkContext->frameIndex]),
      nrdIndirectDiffuseSh0Image(nrdModule->nrdIndirectDiffuseSh0Images_[frameworkContext->frameIndex]),
      nrdIndirectDiffuseSh1Image(nrdModule->nrdIndirectDiffuseSh1Images_[frameworkContext->frameIndex]),
      nrdIndirectSpecularSh0Image(nrdModule->nrdIndirectSpecularSh0Images_[frameworkContext->frameIndex]),
      nrdIndirectSpecularSh1Image(nrdModule->nrdIndirectSpecularSh1Images_[frameworkContext->frameIndex]),
      nrdIndirectMotionVectorImage(nrdModule->nrdIndirectMotionVectorImages_[frameworkContext->frameIndex]),
      nrdIndirectNormalRoughnessImage(nrdModule->nrdIndirectNormalRoughnessImages_[frameworkContext->frameIndex]),
      nrdIndirectLinearDepthImage(nrdModule->nrdIndirectLinearDepthImages_[frameworkContext->frameIndex]),
      nrdIndirectBaseColorMetalnessImage(nrdModule->nrdIndirectBaseColorMetalnessImages_[frameworkContext->frameIndex]),
      nrdDirectMotionVectorImage(nrdModule->nrdDirectMotionVectorImages_[frameworkContext->frameIndex]),
      nrdDirectNormalRoughnessImage(nrdModule->nrdDirectNormalRoughnessImages_[frameworkContext->frameIndex]),
      nrdDirectLinearDepthImage(nrdModule->nrdDirectLinearDepthImages_[frameworkContext->frameIndex]),
      nrdDirectDiffuseRadianceImage(nrdModule->nrdDirectDiffuseRadianceImages_[frameworkContext->frameIndex]),
      nrdDirectSpecularRadianceImage(nrdModule->nrdDirectSpecularRadianceImages_[frameworkContext->frameIndex]),
      nrdDirectDisocclusionThresholdMixImage(
          nrdModule->nrdDirectDisocclusionThresholdMixImages_[frameworkContext->frameIndex]),
      nrdIndirectDisocclusionThresholdMixImage(
          nrdModule->nrdIndirectDisocclusionThresholdMixImages_[frameworkContext->frameIndex]),
      nrdDirectConfidenceImage(nrdModule->nrdDirectConfidenceImages_[frameworkContext->frameIndex]),
      nrdIndirectConfidenceImage(nrdModule->nrdIndirectConfidenceImages_[frameworkContext->frameIndex]),
      denoisedIndirectDiffuseRadianceImage(
          nrdModule->denoisedIndirectDiffuseRadianceImages_[frameworkContext->frameIndex]),
      denoisedIndirectSpecularRadianceImage(
          nrdModule->denoisedIndirectSpecularRadianceImages_[frameworkContext->frameIndex]),
      denoisedIndirectDiffuseSh0Image(nrdModule->denoisedIndirectDiffuseSh0Images_[frameworkContext->frameIndex]),
      denoisedIndirectDiffuseSh1Image(nrdModule->denoisedIndirectDiffuseSh1Images_[frameworkContext->frameIndex]),
      denoisedIndirectSpecularSh0Image(nrdModule->denoisedIndirectSpecularSh0Images_[frameworkContext->frameIndex]),
      denoisedIndirectSpecularSh1Image(nrdModule->denoisedIndirectSpecularSh1Images_[frameworkContext->frameIndex]),
      denoisedDirectDiffuseRadianceImage(nrdModule->denoisedDirectDiffuseRadianceImages_[frameworkContext->frameIndex]),
      denoisedDirectSpecularRadianceImage(
          nrdModule->denoisedDirectSpecularRadianceImages_[frameworkContext->frameIndex]),
      prepareDescriptorTable(nrdModule->prepareDescriptorTables_[frameworkContext->frameIndex]),
      composeDescriptorTable(nrdModule->composeDescriptorTables_[frameworkContext->frameIndex]),
      indirectTexturePool(nrdModule->indirectTexturePools_[frameworkContext->frameIndex]),
      directTexturePool(nrdModule->directTexturePools_[frameworkContext->frameIndex]),
      denoisedRadianceImage(nrdModule->denoisedRadianceImages_[frameworkContext->frameIndex]),
      composedLinearImage(nrdModule->composedLinearImages_[frameworkContext->frameIndex]),
      halfRateIndirectImage(nrdModule->halfRateIndirectImages_[frameworkContext->frameIndex]) {}

void NrdModuleContext::render() {
    MCVR_PROFILE_SCOPE("native.NrdModuleContext.render");
    auto module = nrdModule.lock();
    if (!module || !module->indirectWrapper_) return;

    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto buffers = Renderer::instance().buffers();

    if (!buffers->hasWorldHistory()) {
        module->nrdFrameIndex_ = 0;
        module->hasLastRenderTimestamp_ = false;
        module->halfRateResolveHasHistory_ = false;
    }

    auto worldUBOBuffer = buffers->worldUniformBuffer();
    auto worldUBO = static_cast<vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
    auto lastWorldUBOBuffer = buffers->lastWorldUniformBuffer();
    auto lastWorldUBO = static_cast<vk::Data::WorldUBO *>(lastWorldUBOBuffer->mappedPtr());
    auto skyUBOBuffer = buffers->skyUniformBuffer();

    const auto renderTimestamp = std::chrono::steady_clock::now();
    float frameDeltaMilliseconds = 0.0f;
    float rawFrameDeltaMilliseconds = 0.0f;
    if (module->hasLastRenderTimestamp_) {
        const float frameDeltaSeconds =
            std::chrono::duration<float>(renderTimestamp - module->lastRenderTimestamp_).count();
        if (std::isfinite(frameDeltaSeconds) && frameDeltaSeconds > 0.0f) {
            rawFrameDeltaMilliseconds = frameDeltaSeconds * 1000.0f;
            const float boundedDeltaSeconds = std::clamp(frameDeltaSeconds, 1.0f / 240.0f, 1.0f / 5.0f);
            frameDeltaMilliseconds = boundedDeltaSeconds * 1000.0f;
        }
    }
    module->lastRenderTimestamp_ = renderTimestamp;
    module->hasLastRenderTimestamp_ = true;
    module->updateFrameTime(frameDeltaMilliseconds, module->nrdFrameIndex_ <= 1);
    const char *capture = std::getenv("MCVR_PERFORMANCE_CAPTURE");
    if (capture != nullptr && std::strcmp(capture, "1") == 0) {
        static std::chrono::steady_clock::time_point lastReport{};
        if (renderTimestamp - lastReport >= std::chrono::seconds(5)) {
            const auto &reblur = module->effectiveReblurSettings_;
            const auto &relax = module->effectiveRelaxSettings_;
            std::fprintf(stderr, "[RADIANCE-E15-NRD] frame=%u historyValid=%u raw_frame_ms=%.3f adaptive_fps=%.2f indirect=%ux%u direct=%ux%u reblur_history=%u fast=%u fix=%u relax_diffuse=%u specular=%u separate_direct=%u\n",
                module->nrdFrameIndex_, unsigned(buffers->hasWorldHistory()), rawFrameDeltaMilliseconds,
                module->smoothedFramesPerSecond_, module->indirectWidth_, module->indirectHeight_,
                module->width_, module->height_, reblur.maxAccumulatedFrameNum, reblur.maxFastAccumulatedFrameNum,
                reblur.historyFixFrameNum, relax.diffuseMaxAccumulatedFrameNum, relax.specularMaxAccumulatedFrameNum,
                unsigned(module->isSeparateDirectDenoisingEnabled_));
            lastReport = renderTimestamp;
        }
    }

    glm::mat4 nrdViewToClip = worldUBO->cameraProjMat;
    glm::mat4 nrdViewToClipPrev = lastWorldUBO->cameraProjMat;
    glm::mat4 nrdWorldToViewPrev = lastWorldUBO->cameraEffectedViewMat;
    glm::vec3 cameraDelta = glm::vec3(glm::dvec3(worldUBO->cameraPos) - glm::dvec3(lastWorldUBO->cameraPos));
    nrdWorldToViewPrev[3] += glm::vec4(glm::mat3(lastWorldUBO->cameraEffectedViewMat) * cameraDelta, 0.0f);
    for (int column = 0; column < 4; ++column) {
        nrdViewToClip[column][1] *= -1.0f;
        nrdViewToClipPrev[column][1] *= -1.0f;
    }
    auto makeCommonSettings = [&](uint32_t resourceWidth, uint32_t resourceHeight, bool hasBaseColor) {
        nrd::CommonSettings settings = {};
        std::memcpy(settings.viewToClipMatrix, glm::value_ptr(nrdViewToClip), sizeof(glm::mat4));
        std::memcpy(settings.viewToClipMatrixPrev, glm::value_ptr(nrdViewToClipPrev), sizeof(glm::mat4));
        std::memcpy(settings.worldToViewMatrix, glm::value_ptr(worldUBO->cameraEffectedViewMat), sizeof(glm::mat4));
        std::memcpy(settings.worldToViewMatrixPrev, glm::value_ptr(nrdWorldToViewPrev), sizeof(glm::mat4));
        settings.resourceSize[0] = static_cast<uint16_t>(resourceWidth);
        settings.resourceSize[1] = static_cast<uint16_t>(resourceHeight);
        settings.rectSize[0] = static_cast<uint16_t>(resourceWidth);
        settings.rectSize[1] = static_cast<uint16_t>(resourceHeight);
        settings.resourceSizePrev[0] = static_cast<uint16_t>(resourceWidth);
        settings.resourceSizePrev[1] = static_cast<uint16_t>(resourceHeight);
        settings.rectSizePrev[0] = static_cast<uint16_t>(resourceWidth);
        settings.rectSizePrev[1] = static_cast<uint16_t>(resourceHeight);
        settings.cameraJitter[0] = worldUBO->cameraJitter.x;
        settings.cameraJitter[1] = worldUBO->cameraJitter.y;
        settings.cameraJitterPrev[0] = lastWorldUBO->cameraJitter.x;
        settings.cameraJitterPrev[1] = lastWorldUBO->cameraJitter.y;
        settings.frameIndex = module->nrdFrameIndex_;
        settings.accumulationMode =
            module->nrdFrameIndex_ == 0 ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
        settings.motionVectorScale[0] = 1.0f / static_cast<float>(resourceWidth);
        settings.motionVectorScale[1] = 1.0f / static_cast<float>(resourceHeight);
        settings.motionVectorScale[2] = 1.0f;
        settings.viewZScale = 1.0f;
        settings.denoisingRange = NRD_DENOISING_RANGE;
        settings.timeDeltaBetweenFrames = frameDeltaMilliseconds;
        settings.disocclusionThreshold = 0.01f;
        settings.disocclusionThresholdAlternate = 0.02f;
        settings.isMotionVectorInWorldSpace = false;
        settings.isHistoryConfidenceAvailable = true;
        settings.isBaseColorMetalnessAvailable = hasBaseColor;
        settings.isDisocclusionThresholdMixAvailable = true;
        settings.enableValidation = false;
        return settings;
    };

    nrd::CommonSettings commonSettings = makeCommonSettings(module->indirectWidth_, module->indirectHeight_, true);
    module->indirectWrapper_->setCommonSettings(commonSettings);
    if (module->directWrapper_) {
        nrd::CommonSettings directSettings = makeCommonSettings(module->width_, module->height_, false);
        module->directWrapper_->setCommonSettings(directSettings);
    }

    recordPasses(worldCommandBuffer, worldUBOBuffer, lastWorldUBOBuffer, skyUBOBuffer, module->nrdFrameIndex_++,
                 commonSettings.accumulationMode != nrd::AccumulationMode::CONTINUE);
}

void NrdModuleContext::recordPasses(const std::shared_ptr<vk::CommandBuffer> &worldCommandBuffer,
                                    const std::shared_ptr<vk::HostVisibleBuffer> &worldUBOBuffer,
                                    const std::shared_ptr<vk::HostVisibleBuffer> &lastWorldUBOBuffer,
                                    const std::shared_ptr<vk::HostVisibleBuffer> &skyUBOBuffer,
                                    uint32_t frameIndex,
                                    bool resetHistory) {
    auto module = nrdModule.lock();
    auto framework = frameworkContext.lock()->framework.lock();
    const uint32_t descriptorSlot = frameworkContext.lock()->frameIndex;
    const uint32_t mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    const NrdModePushConstants nrdMode{
        .shEnabled = module->isShEnabled_ ? 1u : 0u,
        .separateDirectEnabled = module->isSeparateDirectDenoisingEnabled_ ? 1u : 0u,
        .frameIndex = frameIndex,
        .nrdMode = module->nrdMode_ == ShaderPackLoader::NrdMode::Checkerboard         ? 1u :
                   module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard ? 2u :
                                                                                         0u,
    };
    const uint32_t dispatchWidth = (module->width_ + 15) / 16;
    const uint32_t dispatchHeight = (module->height_ + 15) / 16;

    auto addBarrier = [&](std::vector<vk::CommandBuffer::ImageMemoryBarrier> &barriers,
                          const std::shared_ptr<vk::DeviceLocalImage> &image) {
        if (image != nullptr) { addComputeImageBarrier(barriers, image, VK_IMAGE_LAYOUT_GENERAL, mainQueueIndex); }
    };
    auto submitBarriers = [&](std::vector<vk::CommandBuffer::ImageMemoryBarrier> &barriers) {
        if (!barriers.empty()) { worldCommandBuffer->barriersBufferImage({}, barriers); }
    };

    if (module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        vk::CommandBuffer::ScopedDebugLabel label(*worldCommandBuffer, "nrd/lighting_response", .3f, .7f, .5f);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(worldCommandBuffer->vkCommandBuffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> outputBarriers;
        addBarrier(outputBarriers, module->lightingResponseImages_[descriptorSlot]);
        addBarrier(outputBarriers, nrdDirectConfidenceImage);
        submitBarriers(outputBarriers);
        auto table = module->lightingResponseTables_[descriptorSlot][module->halfRateResolveParity_];
        const auto &lightingWorld = *static_cast<const vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
        const auto &lightingPreviousWorld = *static_cast<const vk::Data::WorldUBO *>(lastWorldUBOBuffer->mappedPtr());
        const glm::vec2 lightingJitter = lightingWorld.cameraJitter - lightingPreviousWorld.cameraJitter;
        const NrdLightingPushConstants lightingMode{
            module->halfRateResolveHasHistory_ && !resetHistory ? 1u : 0u, lightingJitter.x, lightingJitter.y};
        worldCommandBuffer->bindDescriptorTable(table, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->lightingResponsePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), table->vkPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lightingMode), &lightingMode);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->width_ + 31) / 32, (module->height_ + 31) / 32, 1);
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
        addBarrier(barriers, module->lightingResponseImages_[descriptorSlot]);
        addBarrier(barriers, nrdDirectConfidenceImage);
        submitBarriers(barriers);
    }
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "nrd/prepare", 0.20f, 0.65f, 0.95f);
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;
        addBarrier(imageBarriers, indirectDiffuseRadianceImage);
        addBarrier(imageBarriers, indirectDiffuseDirectionImage);
        addBarrier(imageBarriers, indirectSpecularRadianceImage);
        addBarrier(imageBarriers, indirectSpecularDirectionImage);
        addBarrier(imageBarriers, sharedDiffuseAlbedoMetallicImage);
        addBarrier(imageBarriers, sharedSpecularAlbedoImage);
        addBarrier(imageBarriers, indirectGuideMotionImage);
        addBarrier(imageBarriers, indirectGuideNormalRoughnessImage);
        addBarrier(imageBarriers, indirectGuideDepthImage);
        addBarrier(imageBarriers, composeClearRadianceImage);
        addBarrier(imageBarriers, composeBaseEmissionImage);
        addBarrier(imageBarriers, indirectDiffuseHitDistanceImage);
        addBarrier(imageBarriers, indirectSpecularHitDistanceImage);
        if (module->isSeparateDirectDenoisingEnabled_) {
            addBarrier(imageBarriers, directDiffuseRadianceImage);
            addBarrier(imageBarriers, directSpecularRadianceImage);
            addBarrier(imageBarriers, directHitDistanceImage);
            addBarrier(imageBarriers, directLightGuideNormalRoughnessImage);
            addBarrier(imageBarriers, directLightGuideMotionDepthImage);
            addBarrier(imageBarriers, nrdDirectNormalRoughnessImage);
            addBarrier(imageBarriers, nrdDirectLinearDepthImage);
        }

        if (module->isShEnabled_) {
            addBarrier(imageBarriers, nrdIndirectDiffuseSh0Image);
            addBarrier(imageBarriers, nrdIndirectDiffuseSh1Image);
            addBarrier(imageBarriers, nrdIndirectSpecularSh0Image);
            addBarrier(imageBarriers, nrdIndirectSpecularSh1Image);
        } else {
            addBarrier(imageBarriers, nrdIndirectDiffuseRadianceImage);
            addBarrier(imageBarriers, nrdIndirectSpecularRadianceImage);
        }
        if (module->isSeparateDirectDenoisingEnabled_) {
            addBarrier(imageBarriers, nrdDirectDiffuseRadianceImage);
            addBarrier(imageBarriers, nrdDirectSpecularRadianceImage);
            addBarrier(imageBarriers, nrdDirectDisocclusionThresholdMixImage);
            addBarrier(imageBarriers, nrdDirectConfidenceImage);
            addBarrier(imageBarriers, nrdDirectMotionVectorImage);
            addBarrier(imageBarriers, nrdDirectNormalRoughnessImage);
            addBarrier(imageBarriers, nrdDirectLinearDepthImage);
        }
        addBarrier(imageBarriers, nrdIndirectDisocclusionThresholdMixImage);
        addBarrier(imageBarriers, nrdIndirectConfidenceImage);
        addBarrier(imageBarriers, nrdIndirectMotionVectorImage);
        addBarrier(imageBarriers, nrdIndirectNormalRoughnessImage);
        addBarrier(imageBarriers, nrdIndirectLinearDepthImage);
        addBarrier(imageBarriers, nrdIndirectBaseColorMetalnessImage);
        if (module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
            addBarrier(imageBarriers, module->halfRateDiffuseGuides_[descriptorSlot].motion);
            addBarrier(imageBarriers, module->halfRateDiffuseGuides_[descriptorSlot].normalRoughness);
            addBarrier(imageBarriers, module->halfRateDiffuseGuides_[descriptorSlot].depth);
        }
        submitBarriers(imageBarriers);

        prepareDescriptorTable->bindBuffer(worldUBOBuffer, 2, 0);
        prepareDescriptorTable->bindBuffer(lastWorldUBOBuffer, 2, 1);
        prepareDescriptorTable->bindBuffer(skyUBOBuffer, 2, 2);
        worldCommandBuffer->bindDescriptorTable(prepareDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->preparePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), prepareDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(nrdMode), &nrdMode);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), dispatchWidth, dispatchHeight, 1);
    }

    if (module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
        addBarrier(barriers, composedLinearImage);
        addBarrier(barriers, denoisedRadianceImage);
        addBarrier(barriers, nrdIndirectNormalRoughnessImage);
        addBarrier(barriers, nrdIndirectLinearDepthImage);
        addBarrier(barriers, nrdIndirectMotionVectorImage);
        addBarrier(barriers, module->halfRateDiffuseGuides_[descriptorSlot].motion);
        addBarrier(barriers, module->halfRateDiffuseGuides_[descriptorSlot].normalRoughness);
        addBarrier(barriers, module->halfRateDiffuseGuides_[descriptorSlot].depth);
        submitBarriers(barriers);
    }

    const bool halfRate = module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard;
    const nrd::Identifier reblur = nrd::Identifier(module->isShEnabled_ ? nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH :
                                                                          nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);
    const nrd::Identifier relax = nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR);

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "nrd/indirect_reblur", 0.40f, 0.85f, 0.35f);
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;
        if (module->isShEnabled_) {
            addBarrier(imageBarriers, nrdIndirectDiffuseSh0Image);
            addBarrier(imageBarriers, nrdIndirectDiffuseSh1Image);
            addBarrier(imageBarriers, nrdIndirectSpecularSh0Image);
            addBarrier(imageBarriers, nrdIndirectSpecularSh1Image);
        } else {
            addBarrier(imageBarriers, nrdIndirectDiffuseRadianceImage);
            addBarrier(imageBarriers, nrdIndirectSpecularRadianceImage);
        }
        addBarrier(imageBarriers, nrdIndirectMotionVectorImage);
        addBarrier(imageBarriers, nrdIndirectNormalRoughnessImage);
        addBarrier(imageBarriers, nrdIndirectLinearDepthImage);
        addBarrier(imageBarriers, nrdIndirectBaseColorMetalnessImage);
        addBarrier(imageBarriers, nrdIndirectDisocclusionThresholdMixImage);
        addBarrier(imageBarriers, nrdIndirectConfidenceImage);
        if (halfRate) {
            addBarrier(imageBarriers, module->halfRateDiffuseGuides_[descriptorSlot].motion);
            addBarrier(imageBarriers, module->halfRateDiffuseGuides_[descriptorSlot].normalRoughness);
            addBarrier(imageBarriers, module->halfRateDiffuseGuides_[descriptorSlot].depth);
        }
        submitBarriers(imageBarriers);
        if (halfRate) {
            auto diffusePool = std::make_shared<std::array<std::shared_ptr<vk::DeviceLocalImage>,
                                                          size_t(nrd::ResourceType::MAX_NUM)>>(*indirectTexturePool);
            const auto &guide = module->halfRateDiffuseGuides_[descriptorSlot];
            diffusePool->at(size_t(nrd::ResourceType::IN_MV)) = guide.motion;
            diffusePool->at(size_t(nrd::ResourceType::IN_NORMAL_ROUGHNESS)) = guide.normalRoughness;
            diffusePool->at(size_t(nrd::ResourceType::IN_VIEWZ)) = guide.depth;
            const nrd::Identifier diffuse = nrd::Identifier(module->isShEnabled_ ? nrd::Denoiser::REBLUR_DIFFUSE_SH :
                                                                                nrd::Denoiser::REBLUR_DIFFUSE);
            const nrd::Identifier specular = nrd::Identifier(module->isShEnabled_ ? nrd::Denoiser::REBLUR_SPECULAR_SH :
                                                                                 nrd::Denoiser::REBLUR_SPECULAR);
            module->indirectWrapper_->setUserPoolTexture(diffusePool);
            module->indirectWrapper_->denoise(&diffuse, 1, worldCommandBuffer->vkCommandBuffer());
            module->indirectWrapper_->setUserPoolTexture(indirectTexturePool);
            module->indirectWrapper_->denoise(&specular, 1, worldCommandBuffer->vkCommandBuffer());
        } else {
            module->indirectWrapper_->setUserPoolTexture(indirectTexturePool);
            module->indirectWrapper_->denoise(&reblur, 1, worldCommandBuffer->vkCommandBuffer());
        }
    }

    if (module->isSeparateDirectDenoisingEnabled_) {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "nrd/direct_relax", 0.85f, 0.65f, 0.25f);
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;
        addBarrier(imageBarriers, nrdDirectDiffuseRadianceImage);
        addBarrier(imageBarriers, nrdDirectSpecularRadianceImage);
        addBarrier(imageBarriers, nrdDirectMotionVectorImage);
        addBarrier(imageBarriers, nrdDirectNormalRoughnessImage);
        addBarrier(imageBarriers, nrdDirectLinearDepthImage);
        addBarrier(imageBarriers, nrdDirectDisocclusionThresholdMixImage);
        addBarrier(imageBarriers, nrdDirectConfidenceImage);
        submitBarriers(imageBarriers);
        module->directWrapper_->setUserPoolTexture(directTexturePool);
        module->directWrapper_->denoise(&relax, 1, worldCommandBuffer->vkCommandBuffer());
    }

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "nrd/compose", 0.75f, 0.40f, 0.85f);
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;
        addBarrier(imageBarriers, sharedDiffuseAlbedoMetallicImage);
        addBarrier(imageBarriers, sharedSpecularAlbedoImage);
        addBarrier(imageBarriers, indirectGuideNormalRoughnessImage);
        addBarrier(imageBarriers, indirectGuideDepthImage);
        addBarrier(imageBarriers, composeClearRadianceImage);
        addBarrier(imageBarriers, composeFirstHitRefractionImage);
        addBarrier(imageBarriers, composeBaseEmissionImage);
        addBarrier(imageBarriers, composeFogImage);
        if (module->isSeparateDirectDenoisingEnabled_) {
            addBarrier(imageBarriers, directLightGuideNormalRoughnessImage);
            addBarrier(imageBarriers, directLightGuideMotionDepthImage);
            addBarrier(imageBarriers, nrdDirectNormalRoughnessImage);
            addBarrier(imageBarriers, nrdDirectLinearDepthImage);
        }
        if (module->isShEnabled_) {
            addBarrier(imageBarriers, denoisedIndirectDiffuseSh0Image);
            addBarrier(imageBarriers, denoisedIndirectDiffuseSh1Image);
            addBarrier(imageBarriers, denoisedIndirectSpecularSh0Image);
            addBarrier(imageBarriers, denoisedIndirectSpecularSh1Image);
        } else {
            addBarrier(imageBarriers, denoisedIndirectDiffuseRadianceImage);
            addBarrier(imageBarriers, denoisedIndirectSpecularRadianceImage);
        }
        if (module->isSeparateDirectDenoisingEnabled_) {
            addBarrier(imageBarriers, denoisedDirectDiffuseRadianceImage);
            addBarrier(imageBarriers, denoisedDirectSpecularRadianceImage);
        }
        addBarrier(imageBarriers, denoisedRadianceImage);
        addBarrier(imageBarriers, composedLinearImage);
        if (module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
            addBarrier(imageBarriers, halfRateIndirectImage);
            addBarrier(imageBarriers, module->halfRateSecondaryImages_[descriptorSlot]);
        }
        submitBarriers(imageBarriers);

        composeDescriptorTable->bindBuffer(skyUBOBuffer, 2, 0);
        composeDescriptorTable->bindBuffer(worldUBOBuffer, 2, 1);
        worldCommandBuffer->bindDescriptorTable(composeDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->composePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), composeDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(nrdMode), &nrdMode);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), dispatchWidth, dispatchHeight, 1);
    }
    if (module->nrdMode_ == ShaderPackLoader::NrdMode::HalfRateCheckerboard) {
        const auto &world = *static_cast<const vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
        const auto &previous = *static_cast<const vk::Data::WorldUBO *>(lastWorldUBOBuffer->mappedPtr());
        recordHalfRateResolve(worldCommandBuffer, world, previous, resetHistory, false);
        recordHalfRateResolve(worldCommandBuffer, world, previous, resetHistory, true);
    }
}

void NrdModuleContext::recordHalfRateResolve(const std::shared_ptr<vk::CommandBuffer> &command,
                                            const vk::Data::WorldUBO &world,
                                            const vk::Data::WorldUBO &previousWorld,
                                            bool resetHistory,
                                            bool secondaryPass) {
    auto module = nrdModule.lock();
    if (module->nrdMode_ != ShaderPackLoader::NrdMode::HalfRateCheckerboard) return;

    const std::string label =
        secondaryPass ? "nrd/half_rate/secondary/accumulate" : "nrd/half_rate/indirect/accumulate";
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*command, label.c_str(), 0.70f, 0.45f, 0.85f);

    const auto memoryBarrier = [&] {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(command->vkCommandBuffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };
    memoryBarrier();
    auto &tables = secondaryPass ? module->halfRateSecondaryResolveTables_ : module->halfRateResolveTables_;
    auto table = tables.at(frameworkContext.lock()->frameIndex)[module->halfRateResolveParity_];
    command->bindDescriptorTable(table, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->halfRateResolvePipelines_[secondaryPass ? 1 : 0]);
    const glm::vec2 jitterDelta = world.cameraJitter - previousWorld.cameraJitter;
    NrdResolvePushConstants mode{module->halfRateResolveHasHistory_ && !resetHistory ? 1u : 0u,
                                jitterDelta.x, jitterDelta.y};
    const auto setRays = [&](const vk::Data::WorldUBO &ubo, glm::vec4 &dx, glm::vec4 &dy, glm::vec4 &origin) {
        const glm::vec2 pixelScale = 2.0f / glm::vec2(module->width_, module->height_);
        const glm::vec2 ndc = (glm::vec2(0.5f) + ubo.cameraJitter) * pixelScale - 1.0f;
        const glm::vec4 view = ubo.cameraProjMatInv * glm::vec4(ndc, 0.0f, 1.0f);
        const glm::mat3 toWorld = glm::mat3(ubo.cameraEffectedViewMatInv) / -view.z;
        origin = glm::vec4(toWorld * glm::vec3(view), 0.0f);
        dx = glm::vec4(toWorld * glm::vec3(ubo.cameraProjMatInv[0]) * pixelScale.x, 0.0f);
        dy = glm::vec4(toWorld * glm::vec3(ubo.cameraProjMatInv[1]) * pixelScale.y, 0.0f);
    };
    setRays(world, mode.currentRayX, mode.currentRayY, mode.currentRayOrigin);
    setRays(previousWorld, mode.previousRayX, mode.previousRayY, mode.previousRayOrigin);
    vkCmdPushConstants(command->vkCommandBuffer(), table->vkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(mode), &mode);
    vkCmdDispatch(command->vkCommandBuffer(), (module->width_ + 15) / 16, (module->height_ + 15) / 16, 1);
    if (secondaryPass) {
        memoryBarrier();
        module->halfRateResolveParity_ = 1 - module->halfRateResolveParity_;
        module->halfRateResolveHasHistory_ = true;
    }
}
