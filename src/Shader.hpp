#pragma once
#include "VulkanTypes.hpp"

VkShaderModule load_shader_module(std::string filePath, VkDevice& device);