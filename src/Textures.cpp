#include "Textures.hpp"
#include "Initializers.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

namespace utils {
	bool load_image_from_file(const std::string& filePath, AllocatedImage& outImage, PeteEngine& engine) {
		int texWidth, texHeight, texChannels;

		stbi_uc* pixels = stbi_load(filePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		//void* pixelsPtr = pixels

		if (!pixels) {
			std::cout << "Failed to load texture from: " << filePath << std::endl;
			return false;
		}

		VkDeviceSize imageSize = texWidth * texHeight * 4;

		VkFormat imageFormat = VK_FORMAT_R8G8B8A8_SRGB;

		AllocatedBuffer stagingBuffer = engine.create_buffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_ONLY);

		void* data;
		vmaMapMemory(engine._allocator, stagingBuffer._allocation, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vmaUnmapMemory(engine._allocator, stagingBuffer._allocation);

		stbi_image_free(pixels);

		VkExtent3D imageExtent;
		imageExtent.width = static_cast<uint32_t>(texWidth);
		imageExtent.height = static_cast<uint32_t>(texHeight);
		imageExtent.depth = 1;

		VkImageCreateInfo depthImgInfo = initializers::image_create_info(imageFormat,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent);

		AllocatedImage newImage;
		VmaAllocationCreateInfo depthImgAllocInfo{};
		depthImgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		vmaCreateImage(engine._allocator, &depthImgInfo, &depthImgAllocInfo, &newImage._image,
			&newImage._allocation, nullptr);

		engine.immediate_submit([&](VkCommandBuffer cmd) {
			VkImageSubresourceRange range;
			range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			range.baseMipLevel = 0;
			range.levelCount = 1;
			range.baseArrayLayer = 0;
			range.layerCount = 1;

			VkImageMemoryBarrier imgBarrierToTransfer{};
			imgBarrierToTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imgBarrierToTransfer.pNext = nullptr;

			imgBarrierToTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imgBarrierToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imgBarrierToTransfer.image = newImage._image;
			imgBarrierToTransfer.subresourceRange = range;

			imgBarrierToTransfer.srcAccessMask = 0;
			imgBarrierToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			// barrier image into transfer-receive layout
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 0, nullptr, 1, &imgBarrierToTransfer);

			VkBufferImageCopy copyRegion{};
			copyRegion.bufferOffset = 0;
			copyRegion.bufferRowLength = 0;
			copyRegion.bufferImageHeight = 0;

			copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copyRegion.imageSubresource.mipLevel = 0;
			copyRegion.imageSubresource.baseArrayLayer = 0;
			copyRegion.imageSubresource.layerCount = 1;
			copyRegion.imageExtent = imageExtent;

			vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, newImage._image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

			VkImageMemoryBarrier imgBarrierToReadable = imgBarrierToTransfer;
			imgBarrierToReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imgBarrierToReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			imgBarrierToReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imgBarrierToReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0, 0, nullptr, 0, nullptr, 1, &imgBarrierToReadable);
			});

		engine._deletionQueue.push_function([=]() {
			vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
			});

		vmaDestroyBuffer(engine._allocator, stagingBuffer._buffer, stagingBuffer._allocation);

		outImage = newImage;

		std::cout << "succesfully loaded texture: " << filePath << std::endl;
		return true;
	}
}