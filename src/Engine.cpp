#include "Engine.hpp"
#include "VulkanTypes.hpp"
#include "Initializers.hpp"
#include "Pipeline.hpp"
#include "Shader.hpp"
#include "VkBootstrap.h"
#include <GLFW/glfw3.h>
#include <glm/gtx/transform.hpp>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <iostream>
#include <fstream>

#define VK_CHECK(x)                                             \
	do {														\
		VkResult err = x;                                       \
		if (err) {                                              \
			std::cout << "Vulkan error: " << err << std::endl;	\
			abort();                                            \
		}                                                       \
	} while (0);

void PeteEngine::init() {

	init_glfw();
	init_vulkan();
	init_swapchain();
	init_commands();
	init_default_renderpass();
	init_framebuffers();
	init_sync_structures();
	init_pipelines();

	load_meshes();

	_isInitialized = true;
}

void PeteEngine::init_glfw() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	_window = glfwCreateWindow(_windowExtent.width, _windowExtent.height, "PeteEngine v0.0.0", nullptr, nullptr);
	if (_window == nullptr)
		throw std::runtime_error("Failed to create window!");

	glfwSetWindowUserPointer(_window, this);
	glfwSetFramebufferSizeCallback(_window, framebuffer_resize_callback);

	glfwSetKeyCallback(_window, key_callback);
}

void PeteEngine::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	PeteEngine *inst = static_cast<PeteEngine *>(glfwGetWindowUserPointer(window));
	
	if (key == GLFW_KEY_E && action == GLFW_PRESS)
		// cycle between 0 and 1
		inst->_selectedShader = (inst->_selectedShader + 1) % 2;
}

void PeteEngine::init_vulkan() {
	vkb::InstanceBuilder builder;

	// build a Vulkan instance with validation layers turned on
	builder.set_app_name("PeteEngine")
		.require_api_version(1, 2, 0);

#ifndef NDEBUG
	builder.request_validation_layers(true)
		.use_default_debug_messenger();
#endif

	vkb::Instance vkbInst = builder.build().value();

	_instance = vkbInst.instance;
	_debugMessenger = vkbInst.debug_messenger;

	// create a surface for the instance to render to
	glfwCreateWindowSurface(_instance, _window, nullptr, &_surface);

	// select a GPU that fits our needs
	vkb::PhysicalDeviceSelector selector{ vkbInst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 2)
		.set_surface(_surface)
		.select()
		.value();

	// create the logical device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	vkb::Device vkbDevice = deviceBuilder
		.build()
		.value();

	_device = vkbDevice.device;
	_physDevice = physicalDevice.physical_device;

	// get a graphics queue from the device
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	// create the memory allocator
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = _physDevice;
	allocatorInfo.device = _device;
	allocatorInfo.instance = _instance;
	vmaCreateAllocator(&allocatorInfo, &_allocator);

	_deletionQueue.push_function([=]() {
		vmaDestroyAllocator(_allocator);
	});
}

void PeteEngine::init_swapchain() {
	vkb::SwapchainBuilder swapchainBuilder{ _physDevice, _device, _surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();
	_swapchainImageFormat = vkbSwapchain.image_format;

	_deletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
	});
}

void PeteEngine::init_commands() {
	// create command pool
	VkCommandPoolCreateInfo commandPoolInfo = initializers::command_pool_create_info(
		_graphicsQueueFamily,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT // allow resetting of individual command buffers
		);

	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

	// create the command buffer
	VkCommandBufferAllocateInfo commandBufferAllocInfo = initializers::command_buffer_allocate_info(
		_commandPool,
		1
	);

	VK_CHECK(vkAllocateCommandBuffers(_device, &commandBufferAllocInfo, &_mainCommandBuffer));

	_deletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _commandPool, nullptr);
	});
}

void PeteEngine::init_default_renderpass() {
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = _swapchainImageFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;

	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	VK_CHECK(vkCreateRenderPass(_device, &renderPassInfo, nullptr, &_renderPass));

	_deletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
	});
}

void PeteEngine::init_framebuffers() {
	VkFramebufferCreateInfo frameBufferInfo{};
	frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferInfo.pNext = nullptr;

	frameBufferInfo.renderPass = _renderPass;
	frameBufferInfo.attachmentCount = 1;
	frameBufferInfo.width = _windowExtent.width;
	frameBufferInfo.height = _windowExtent.height;
	frameBufferInfo.layers = 1;

	const size_t swapchainImageCount = _swapchainImages.size();
	_framebuffers = std::vector<VkFramebuffer>(swapchainImageCount);

	// create framebuffer for every image view in the swapchain
	for (size_t i = 0; i < swapchainImageCount; i++) {
		frameBufferInfo.pAttachments = &_swapchainImageViews[i];
		VK_CHECK(vkCreateFramebuffer(_device, &frameBufferInfo, nullptr, &_framebuffers[i]))

		_deletionQueue.push_function([=]() {
		vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
		vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		});
	}
}

void PeteEngine::init_sync_structures() {
	VkFenceCreateInfo fenceCreateInfo = initializers::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

	_deletionQueue.push_function([=]() {
		vkDestroyFence(_device, _renderFence, nullptr);
	});

	VkSemaphoreCreateInfo semaphoreCreateInfo = initializers::semaphore_create_info();

	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));

	_deletionQueue.push_function([=]() {
		vkDestroySemaphore(_device, _presentSemaphore, nullptr);
		vkDestroySemaphore(_device, _renderSemaphore, nullptr);
	});
}

void PeteEngine::init_pipelines() {
	VkShaderModule meshVertShader = load_shader_module("shaders/tri_mesh_vert.spv", _device);
	VkShaderModule meshFragShader = load_shader_module("shaders/frag.spv", _device);

	// create the pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = initializers::pipeline_layout_create_info();

	VkPushConstantRange pushConstant;
	pushConstant.offset = 0;
	pushConstant.size = sizeof(MeshPushConstants);
	pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

	VK_CHECK(vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &_meshPipelineLayout));

	// build the mesh pipeline
	PipelineBuilder pipelineBuilder;

	pipelineBuilder._shaderStages.push_back(
		initializers::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader)
	);

	pipelineBuilder._shaderStages.push_back(
		initializers::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, meshFragShader)
	);

	pipelineBuilder._viewport.x = 0.0f;
	pipelineBuilder._viewport.y = 0.0f;
	pipelineBuilder._viewport.width = static_cast<float>(_windowExtent.width);
	pipelineBuilder._viewport.height = static_cast<float>(_windowExtent.height);
	pipelineBuilder._viewport.minDepth = 0.0f;
	pipelineBuilder._viewport.maxDepth = 1.0f;

	pipelineBuilder._scissor.offset = { 0, 0 };
	pipelineBuilder._scissor.extent = _windowExtent;

	pipelineBuilder._vertexInputInfo = initializers::vertex_input_state_create_info();
	pipelineBuilder._inputAssembly = initializers::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	VertexInputDescription vertexDescription = Vertex::get_vertex_description();

	pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();

	pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();
	pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();

	// draw filled triangles
	pipelineBuilder._rasterizer = initializers::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	
	pipelineBuilder._multisampling = initializers::multisampling_state_create_info();
	pipelineBuilder._colorBlendAttachment = initializers::color_blend_attachment_state();
	pipelineBuilder._pipelineLayout = _meshPipelineLayout;

	_meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

	// destroy shader modules now that the pipeline has been created
	vkDestroyShaderModule(_device, meshVertShader, nullptr);
	vkDestroyShaderModule(_device, meshFragShader, nullptr);

	// queue destruction of the pipelines and their layout
	_deletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, _meshPipeline, nullptr);

		vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
	});
}

void PeteEngine::cleanup() {
	if (_isInitialized) {

		vkWaitForFences(_device, 1, &_renderFence, true, 1000000000);

		_deletionQueue.flush();

		vkDestroySurfaceKHR(_instance, _surface, nullptr);
		vkDestroyDevice(_device, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
		vkDestroyInstance(_instance, nullptr);

		glfwDestroyWindow(_window);
		glfwTerminate();
	}
}

void PeteEngine::draw() {
	// wait until the GPU has finished rendering the last frame
	VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &_renderFence));

	// request image from swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapchainImageIndex));

	VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

	VkCommandBuffer cmd = _mainCommandBuffer;

	// begin the command buffer recording.
	VkCommandBufferBeginInfo cmdBeginInfo{};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;

	cmdBeginInfo.pInheritanceInfo = nullptr;
	// use the command buffer once
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	VkClearValue clearValue;
	float flash = abs(sin(_frameNumber / 60.f));
	clearValue.color = { { flash * 0.5f + 0.5f, 1.0f - flash, flash, 1.0f } };

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.pNext = nullptr;

	renderPassInfo.renderPass = _renderPass;
	renderPassInfo.renderArea.offset.x = 0;
	renderPassInfo.renderArea.offset.y = 0;
	renderPassInfo.renderArea.extent = _windowExtent;
	renderPassInfo.framebuffer = _framebuffers[swapchainImageIndex];

	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearValue;

	// begin the render pass
	vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

	// TODO: funny camera
	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &_monkeyMesh._vertexBuffer._buffer, &offset);

	glm::vec3 camPos = { 0.0f, 0.0f, -3.0f };
	glm::mat4 view = glm::translate(glm::mat4(1.0f), camPos);

	// camera projection
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1600.0f / 900.0f, 0.1f, 200.0f);
	projection[1][1] *= -1;

	// rotate model
	//glm::mat4 model = glm::mat4(1.0f);
	glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(_frameNumber * 0.5f), glm::vec3(1, 0, 0));

	glm::mat4 meshMatrix = projection * view * model;

	MeshPushConstants constants;
	constants.renderMatrix = meshMatrix;

	vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

	vkCmdDraw(cmd, _monkeyMesh._vertices.size(), 1, 0, 0);

	// transition render pass to finalized state
	vkCmdEndRenderPass(cmd);

	// transition command buffer to executable state
	VK_CHECK(vkEndCommandBuffer(cmd));

	// start submitting to GPU
	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &_presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &_renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

	// start presenting to screen
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &_renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	_frameNumber++;
}

void PeteEngine::run() {
	while (!glfwWindowShouldClose(_window))
	{
		glfwPollEvents();

		draw();
	}
}

void PeteEngine::load_meshes()
{
	_triangleMesh._vertices.resize(3);

	_triangleMesh._vertices[0].position = { 1.f, 1.f, 0.0f };
	_triangleMesh._vertices[1].position = { -1.f, 1.f, 0.0f };
	_triangleMesh._vertices[2].position = { 0.f,-1.f, 0.0f };

	_triangleMesh._vertices[0].color = { 0.f, 1.f, 0.0f }; // pure green
	_triangleMesh._vertices[1].color = { 1.f, 1.f, 0.0f };
	_triangleMesh._vertices[2].color = { 0.f, 1.f, 1.0f };

	_monkeyMesh.load_from_obj("assets/monkey_smooth.obj");

	upload_mesh(_triangleMesh);
	upload_mesh(_monkeyMesh);
}

void PeteEngine::upload_mesh(Mesh& mesh) {
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = mesh._vertices.size() * sizeof(Vertex);
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
		&mesh._vertexBuffer._buffer,
		&mesh._vertexBuffer._allocation,
		nullptr)
	);

	_deletionQueue.push_function([=]() {
		vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
	});

	void* data;
	vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);

	memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));

	vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}

void PeteEngine::framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
	// GLFW does not know how to properly call a member function
	auto currentWindow = reinterpret_cast<PeteEngine*>(glfwGetWindowUserPointer(window));
	currentWindow->_framebufferResized = true;
}

void DeletionQueue::push_function(std::function<void()>&& function) {
	deletors.push_back(function);
}

void DeletionQueue::flush() {
	for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
		(*it)(); // call deletion function
	}

	deletors.clear();
}
