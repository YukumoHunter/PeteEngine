#pragma once
#include "VulkanTypes.hpp"

VkShaderModule load_shader_module(const char* filePath, VkDevice& device);