#pragma once

#include "VulkanTypes.hpp"
#include "Engine.hpp"

namespace utils {
	bool load_image_from_file(const std::string& filePath, AllocatedImage& outImage, PeteEngine& engine);
}