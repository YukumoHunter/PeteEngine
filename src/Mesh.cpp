#include "Mesh.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <iostream>
#include <string>

VertexInputDescription Vertex::get_vertex_description()
{
	VertexInputDescription description;

	// single vertex buffer binding, with a per-vertex rate
	VkVertexInputBindingDescription mainBinding{};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	//Position will be stored at Location 0
	VkVertexInputAttributeDescription positionAttribute{};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = offsetof(Vertex, position);

	//Normal will be stored at Location 1
	VkVertexInputAttributeDescription normalAttribute{};
	normalAttribute.binding = 0;
	normalAttribute.location = 1;
	normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset = offsetof(Vertex, normal);

	//Color will be stored at Location 2
	VkVertexInputAttributeDescription colorAttribute{};
	colorAttribute.binding = 0;
	colorAttribute.location = 2;
	colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset = offsetof(Vertex, color);

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(colorAttribute);

	return description;
}

void Mesh::load_from_obj(std::string filePath) {
	// contains the vertex arrays
	tinyobj::attrib_t attrib;
	// contains info for each object in file
	std::vector<tinyobj::shape_t> shapes;
	// contains material info of each shape
	std::vector<tinyobj::material_t> materials;

	std::string warn, err;

	std::string baseDir;

	// get dir of file
	auto cutoff = filePath.rfind('/');
	if (cutoff != std::string::npos) {
		baseDir = filePath.substr(0, cutoff);
	}

	// tinyobj please just look for the mtl in the same dir as filePath why you do this
	if (baseDir.empty())
		tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str(), nullptr);
	else
		tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str(), baseDir.c_str());

	if (!warn.empty())
		std::cout << "Warning: " << warn << std::endl;

	if (!err.empty())
		throw std::runtime_error("Failed to load obj: " + err);

	for (size_t s = 0; s < shapes.size(); s++) {
		size_t indexOffset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			// hardcoded 3 vertices in a triangle
			int fv = 3;

			for (size_t v = 0; v < fv; v++) {
				// access the vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[indexOffset + v];
				// vertex position
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
				// vertex normals
				tinyobj::real_t nx = attrib.normals[3 * idx.vertex_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.vertex_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.vertex_index + 2];

				// create new vertex
				Vertex vert;
				vert.position.x = vx;
				vert.position.y = vy;
				vert.position.z = vz;

				vert.normal.x = nx;
				vert.normal.y = ny;
				vert.normal.z = nz;

				vert.color = vert.normal;

				_vertices.push_back(vert);
			}
			indexOffset += fv;
		}
	}
}