#pragma once
#include "VulkanTypes.hpp"

VkShaderModule load_shader_module(const std::string& filePath, VkDevice& device);