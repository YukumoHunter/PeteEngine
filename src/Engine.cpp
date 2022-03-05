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
	init_descriptors();
	init_pipelines();
	init_scene();

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
	
	if (key == GLFW_KEY_E && action == GLFW_REPEAT) {
		// cycle between 0 and 1
		auto& renderables = inst->_renderables;
		for (int i = 0; i < renderables.size(); i++) {
			renderables[i].transformMatrix = glm::translate(renderables[i].transformMatrix, glm::vec3(0, 0, .5f));
		}
	}

	if (key == GLFW_KEY_R && action == GLFW_REPEAT) {
		// cycle between 0 and 1
		auto& renderables = inst->_renderables;
		for (int i = 0; i < renderables.size(); i++) {
			renderables[i].transformMatrix = glm::translate(renderables[i].transformMatrix, glm::vec3(0, 0, -.5f));
		}
	}
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

	_gpuProperties = vkbDevice.physical_device.properties;
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

	// depth image matches window extent
	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};

	_depthFormat = VK_FORMAT_D32_SFLOAT;
	
	VkImageCreateInfo depthImageInfo = initializers::image_create_info(_depthFormat,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	VmaAllocationCreateInfo depthImageAllocInfo{};
	depthImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	depthImageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(_allocator, &depthImageInfo, &depthImageAllocInfo, &_depthImage._image,
		&_depthImage._allocation, nullptr);
	
	VkImageViewCreateInfo depthImageViewInfo = initializers::image_view_create_info(_depthFormat,
		_depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(_device, &depthImageViewInfo, nullptr, &_depthImageView));

	_deletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthImageView, nullptr);
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
	});
}

void PeteEngine::init_commands() {
	// create command pool
	VkCommandPoolCreateInfo commandPoolInfo = initializers::command_pool_create_info(
		_graphicsQueueFamily,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT // allow resetting of individual command buffers
	);

	for (int i = 0; i < FRAME_OVERLAP; i++) {

		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

		//allocate the default command buffer that we will use for rendering
		VkCommandBufferAllocateInfo commandBufferAllocInfo = initializers::command_buffer_allocate_info(
			_frames[i]._commandPool,
			1
		);

		VK_CHECK(vkAllocateCommandBuffers(_device, &commandBufferAllocInfo, &_frames[i]._mainCommandBuffer));

		_deletionQueue.push_function([=]() {
			vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
		});
	}
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

	// depth image for depth testing
	VkAttachmentDescription depthAttachment{};
	depthAttachment.format = _depthFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentRef{};
	depthAttachmentRef.attachment = 1;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	// subpass dependent on previous render pass
	VkSubpassDependency depthDependency{};
	depthDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	depthDependency.dstSubpass = 0;
	depthDependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depthDependency.srcAccessMask = 0;
	depthDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depthDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;


	VkAttachmentDescription attachments[] = { colorAttachment, depthAttachment };
	VkSubpassDependency dependencies[] = { dependency, depthDependency };

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	renderPassInfo.attachmentCount = 2;
	renderPassInfo.pAttachments = attachments;

	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	renderPassInfo.dependencyCount = 2;
	renderPassInfo.pDependencies = dependencies;

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

		VkImageView attachments[2];
		attachments[0] = _swapchainImageViews[i];
		attachments[1] = _depthImageView;

		frameBufferInfo.pAttachments = attachments;
		frameBufferInfo.attachmentCount = 2;

		VK_CHECK(vkCreateFramebuffer(_device, &frameBufferInfo, nullptr, &_framebuffers[i]))

		_deletionQueue.push_function([=]() {
		vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
		vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		});
	}
}

void PeteEngine::init_sync_structures() {
	VkFenceCreateInfo fenceCreateInfo = initializers::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphoreCreateInfo = initializers::semaphore_create_info();

	for (int i = 0; i < FRAME_OVERLAP; i++) {
		VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

		_deletionQueue.push_function([=]() {
			vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
		});

		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));
		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

		_deletionQueue.push_function([=]() {
			vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
			vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
		});
	}
}

void PeteEngine::init_descriptors() {
	// create descriptor pool that will hold 10 uniform buffers
	std::vector<VkDescriptorPoolSize> sizes = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 }
	};

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.pNext = nullptr;

	poolInfo.maxSets = 10;
	poolInfo.poolSizeCount = static_cast<uint32_t>(sizes.size());
	poolInfo.pPoolSizes = sizes.data();

	vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool);
	
	// buffer for camera
	VkDescriptorSetLayoutBinding cameraBufferBinding = initializers::descriptor_set_layout_binding(
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

	// buffer for scene
	VkDescriptorSetLayoutBinding sceneBufferBinding = initializers::descriptor_set_layout_binding(
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);

	VkDescriptorSetLayoutBinding bufferBindings[] = { cameraBufferBinding, sceneBufferBinding };

	VkDescriptorSetLayoutCreateInfo setInfo{};
	setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.pNext = nullptr;

	setInfo.bindingCount = 2;
	setInfo.pBindings = bufferBindings;
	setInfo.flags = 0;

	vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);
	
	const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));
	_sceneParameterBuffer = create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// create camera buffers
	for (int i = 0; i < FRAME_OVERLAP; i++) {
		_frames[i].cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.pNext = nullptr;
		
		// using pool we just created
		allocInfo.descriptorPool = _descriptorPool;
		allocInfo.descriptorSetCount = 1;
		// using the global layout for descriptor sets
		allocInfo.pSetLayouts = &_globalSetLayout;

		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].globalDescriptor);
		
		// now that the descriptor set is allocated point it to the the buffer
		VkDescriptorBufferInfo cameraBufferInfo{};
		cameraBufferInfo.buffer = _frames[i].cameraBuffer._buffer;
		cameraBufferInfo.offset = 0;
		cameraBufferInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo sceneBufferInfo{};
		sceneBufferInfo.buffer = _sceneParameterBuffer._buffer;
		sceneBufferInfo.offset = 0;
		sceneBufferInfo.range = sizeof(GPUSceneData);

		VkWriteDescriptorSet camWrite = initializers::write_descriptor_buffer(
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &cameraBufferInfo, 0);

		VkWriteDescriptorSet sceneWrite = initializers::write_descriptor_buffer(
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, _frames[i].globalDescriptor, &sceneBufferInfo, 1);

		VkWriteDescriptorSet setWrites[] = { camWrite, sceneWrite };
		
		// point at it...NOW!
		vkUpdateDescriptorSets(_device, 2, setWrites, 0, nullptr);
	}

	for (int i = 0; i < FRAME_OVERLAP; i++) {
		_deletionQueue.push_function([=]() {
			vmaDestroyBuffer(_allocator, _frames[i].cameraBuffer._buffer, _frames[i].cameraBuffer._allocation);
		});
	}

	// don't forget to delete descriptor pool/set layout
	_deletionQueue.push_function([=]() {
		vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);

		vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
	});

}

void PeteEngine::init_pipelines() {
	VkShaderModule meshVertShader = load_shader_module("shaders/vert.spv", _device);
	VkShaderModule meshFragShader = load_shader_module("shaders/frag.spv", _device);

	VkPushConstantRange pushConstant;
	pushConstant.offset = 0;
	pushConstant.size = sizeof(MeshPushConstants);
	pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	// create the pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = initializers::pipeline_layout_create_info();


	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &_globalSetLayout;

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

	pipelineBuilder._colorBlendAttachment = initializers::color_blend_attachment_state();
	pipelineBuilder._depthStencil = initializers::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
	pipelineBuilder._multisampling = initializers::multisampling_state_create_info();
	pipelineBuilder._pipelineLayout = _meshPipelineLayout;

	_meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

	// create a default material
	create_material(_meshPipeline, _meshPipelineLayout, "default");

	// destroy shader modules now that the pipeline has been created
	vkDestroyShaderModule(_device, meshVertShader, nullptr);
	vkDestroyShaderModule(_device, meshFragShader, nullptr);

	// queue destruction of the pipelines and their layout
	_deletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, _meshPipeline, nullptr);

		vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
	});
}

void PeteEngine::init_scene() {
	load_mesh("assets/monkey_smooth.obj", "monkey");

	for (int x = 0; x < 4; x++) {
		for (int y = 0; y < 4; y++) {
			RenderObject monkey{};
			monkey.mesh = get_mesh("monkey");
			monkey.material = get_material("default");
			monkey.transformMatrix = glm::translate(
				glm::scale(glm::mat4(1.0f), glm::vec3(0.5f)),
				glm::vec3(x * 5 - 7.5f, y * 5 - 7.5f, 0.0f)
			);

			_renderables.push_back(monkey);
		}
	}
}

void PeteEngine::load_mesh(const std::string& filePath, const std::string& name)
{
	Mesh mesh;
	mesh.load_from_obj(filePath);
	upload_mesh(mesh);

	_meshes[name] = mesh;
}

void PeteEngine::cleanup() {
	if (_isInitialized) {

		// wait until all frames have finished rendering
		VkFence fences[FRAME_OVERLAP];
		for (int i = 0; i < FRAME_OVERLAP; i++) { fences[i] = _frames[i]._renderFence; }
		vkWaitForFences(_device, 2, fences, true, 1000000000);

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
	auto currentFrame = get_current_frame();

	VK_CHECK(vkWaitForFences(_device, 1, &currentFrame._renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &currentFrame._renderFence));

	// request image from swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, currentFrame._presentSemaphore, nullptr, &swapchainImageIndex));

	VK_CHECK(vkResetCommandBuffer(currentFrame._mainCommandBuffer, 0));
	VkCommandBuffer cmd = currentFrame._mainCommandBuffer;

	// begin the command buffer recording.
	VkCommandBufferBeginInfo cmdBeginInfo{};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;

	cmdBeginInfo.pInheritanceInfo = nullptr;
	// use the command buffer once
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	VkClearValue clearValue;
	clearValue.color = { 1.0f, 1.0f, 1.0f };

	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.0f;

	VkClearValue clearValues[] = { clearValue, depthClear };

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.pNext = nullptr;

	renderPassInfo.renderPass = _renderPass;
	renderPassInfo.renderArea.offset.x = 0;
	renderPassInfo.renderArea.offset.y = 0;
	renderPassInfo.renderArea.extent = _windowExtent;
	renderPassInfo.framebuffer = _framebuffers[swapchainImageIndex];

	renderPassInfo.clearValueCount = 2;
	renderPassInfo.pClearValues = clearValues;

	// begin the render pass
	vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_objects(cmd, _renderables);

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
	submit.pWaitSemaphores = &currentFrame._presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &currentFrame._renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, currentFrame._renderFence));

	// start presenting to screen
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &currentFrame._renderSemaphore;
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

Material& PeteEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name) {
	Material mat;
	mat.pipeline = pipeline;
	mat.pipelineLayout = layout;
	_materials[name] = mat;
	return _materials[name];
}

Material* PeteEngine::get_material(const std::string& name) {
	auto material = _materials.find(name);
	if (material == _materials.end())
		return nullptr;
	else
		return &material->second;
}

Mesh* PeteEngine::get_mesh(const std::string& name) {
	auto mesh = _meshes.find(name);
	if (mesh == _meshes.end())
		return nullptr;
	else
		return &mesh->second;
}

FrameData& PeteEngine::get_current_frame() {
	return _frames[_frameNumber % FRAME_OVERLAP];
}

void PeteEngine::draw_objects(VkCommandBuffer cmd, std::vector<RenderObject>& renderables) {
	glm::vec3 camPos = { 0.0f, 0.0f, -10.0f };
	glm::mat4 view = glm::translate(glm::mat4(1.0f), camPos);
	glm::mat4 projection = glm::perspective(glm::radians(85.0f), 1600.0f / 900.0f, 0.1f, 200.0f);
	projection[1][1] *= -1;

	GPUCameraData cameraData{};
	cameraData.projection = projection;
	cameraData.view = view;
	cameraData.viewProjection = projection * view;

	// copy camera data to the buffer on the GPU
	auto currentFrame = get_current_frame();
	void* data;
	vmaMapMemory(_allocator, currentFrame.cameraBuffer._allocation, &data);
	memcpy(data, &cameraData, sizeof(GPUCameraData));
	vmaUnmapMemory(_allocator, currentFrame.cameraBuffer._allocation);

	// copy scene data too
	float framed = (_frameNumber / 120.f);
	_sceneParameters.ambientColor = { sin(framed),0,cos(framed),1 };
	char* sceneData;
	vmaMapMemory(_allocator, _sceneParameterBuffer._allocation, (void**)&sceneData);
	int frameIndex = _frameNumber % FRAME_OVERLAP;
	sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
	memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));
	vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;
	for (int i = 0; i < renderables.size(); i++) {
		RenderObject& object = renderables[i];

		// prevent rebinding if materials are the same
		if (object.material != lastMaterial) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
			lastMaterial = object.material;

			uint32_t uniformOffset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout,
				0, 1, &currentFrame.globalDescriptor, 1, &uniformOffset);
		}

		glm::mat4 model = object.transformMatrix;
		model = glm::rotate(model, (360.0f * i / 4 + _frameNumber) / 72, glm::vec3(1.0f));
		auto meshMatrix = model;

		MeshPushConstants constants;
		constants.renderMatrix = meshMatrix;

		vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
			sizeof(MeshPushConstants), &constants);

		// prevent rebinding if meshes are the same
		if (object.mesh != lastMesh) {
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
			lastMesh = object.mesh;
		}

		vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, 0);
	}
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

AllocatedBuffer PeteEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;

	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer;

	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
		&newBuffer._buffer,
		&newBuffer._allocation,
		nullptr
	));

	return newBuffer;
}

size_t PeteEngine::pad_uniform_buffer_size(size_t originalSize) {
	// calculate the required alignment based on min offset on physDevice
	// https://github.com/SaschaWillems/Vulkan/blob/master/examples/dynamicuniformbuffer/dynamicuniformbuffer.cpp
	size_t minAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;

	if (minAlignment > 0)
		alignedSize = (alignedSize + minAlignment - 1) & ~(minAlignment - 1);

	return alignedSize;
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
