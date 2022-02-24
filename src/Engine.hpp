#pragma once
#include <functional>
#include <vector>
#include <glm/glm.hpp>
#include "Mesh.hpp"
#include "VulkanTypes.hpp"

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

class PeteEngine {
public:
	bool _isInitialized{ false };
	bool _framebufferResized{ false };
	int _frameNumber{ 0 };
	int _selectedShader{ 0 };

	struct GLFWwindow* _window{ nullptr };
	VkExtent2D _windowExtent{ 1920 , 1080 };

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

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkRenderPass _renderPass;

	std::vector<VkFramebuffer> _framebuffers;

	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkPipelineLayout _meshPipelineLayout;

	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;

	DeletionQueue _deletionQueue;

	void init();
	void cleanup();
	void draw();
	void run();

private:
	void init_glfw();
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_default_renderpass();
	void init_framebuffers();
	void init_sync_structures();
	void init_pipelines();

	void load_meshes();
	void upload_mesh(Mesh& mesh);

	static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
};
