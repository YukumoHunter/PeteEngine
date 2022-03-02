#pragma once
#include <functional>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include "Mesh.hpp"
#include "VulkanTypes.hpp"

const int FRAME_OVERLAP = 2;

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 renderMatrix;
};

struct DeletionQueue
{
	std::vector<std::function<void()>> deletors;

	// push destruction functions to the queue
	void push_function(std::function<void()>&& function);
	void flush();
};

struct GPUCameraData {
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewProjection;
};

struct FrameData {
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	AllocatedBuffer cameraBuffer;

	VkDescriptorSet globalDescriptor;
};

class PeteEngine {
public:
	bool _isInitialized{ false };
	bool _framebufferResized{ false };
	int _frameNumber{ 0 };

	struct GLFWwindow* _window{ nullptr };
	VkExtent2D _windowExtent{ 1920 , 1080 };


	std::vector<RenderObject> _renderables;
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	VmaAllocator _allocator;

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debugMessenger;
	VkPhysicalDevice _physDevice;
	VkDevice _device;
	VkSurfaceKHR _surface;

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkRenderPass _renderPass;
	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	VkFormat _depthFormat;

	std::vector<VkFramebuffer> _framebuffers;

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	FrameData _frames[FRAME_OVERLAP];

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorPool _descriptorPool;

	DeletionQueue _deletionQueue;

	void init();
	void cleanup();
	void draw();
	void run();

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	Material& create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);

	Material* get_material(const std::string& name);
	Mesh* get_mesh(const std::string& name);

	// draw function
	void draw_objects(VkCommandBuffer cmd, std::vector<RenderObject>& renderables);
	
	FrameData& get_current_frame();

private:
	void init_glfw();
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_default_renderpass();
	void init_framebuffers();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_scene();

	void load_mesh(const std::string& filePath, const std::string& name);
	void upload_mesh(Mesh& mesh);

	static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
};
