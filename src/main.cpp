#include <gp_system.h>

#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/camera.h"
#include <imgui.h>
#include <thread>

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::batch_renderer_2d renderer_2d(resources, global_matrices);
blt::gfx::first_person_camera camera;

std::vector<blt::gfx::texture_gl2D*> gl_images;
blt::size_t population_size = 64;

namespace im = ImGui;

std::atomic_bool run_generation = false;
std::atomic_bool should_exit = false;

std::thread run_gp()
{
	return std::thread{
		[]() {
			setup_gp_system(population_size);
			while (!should_terminate() && !should_exit)
			{
				if (run_generation)
				{
					run_step();
					run_generation = false;
				} else
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			cleanup();
		}
	};
}

void init(const blt::gfx::window_data&)
{
	using namespace blt::gfx;

	for (blt::size_t i = 0; i < population_size; i++)
	{
		auto texture = new texture_gl2D(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGBA8);
		gl_images.push_back(texture);
		resources.set(std::to_string(i), texture);
	}
	global_matrices.create_internals();
	resources.load_resources();
	renderer_2d.create();

	setWindowSize(1400, 720);
}

void update(const blt::gfx::window_data& data)
{
	global_matrices.update_perspectives(data.width, data.height, 90, 0.1, 2000);

	camera.update();
	camera.update_view(global_matrices);
	global_matrices.update();

	im::ShowDemoWindow();

	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y));
	ImGui::Begin("MainWindow", nullptr,
				ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
				ImGuiWindowFlags_NoBringToFrontOnFocus);

	// Create the tab bar
	if (ImGui::BeginTabBar("MainTabs"))
	{
		// 1. Run GP tab
		if (ImGui::BeginTabItem("Run GP"))
		{
			// Left child - fixed width (250px)
			ImGui::BeginChild("ControlPanel", ImVec2(250, 0), true);
			{
				ImGui::Text("Control Panel");
				ImGui::Separator();
				if (ImGui::Button("Run Step"))
				{
					BLT_TRACE("Running step");
					run_generation = true;
				}
			}
			ImGui::EndChild();

			// Right child - take the remaining space
			ImGui::SameLine();
			// ImGui::BeginChild("MainContent", ImVec2(0, 0), false, ImGuiWindowFlags_NoBackground);
			{}
			// ImGui::EndChild();

			ImGui::EndTabItem();
		}

		// 2. Explore Trees tab
		if (ImGui::BeginTabItem("Explore Trees"))
		{
			ImGui::Text("Here you can explore trees.");
			// Additional UI for exploring trees
			ImGui::EndTabItem();
		}

		// 3. Statistics tab
		if (ImGui::BeginTabItem("Statistics"))
		{
			ImGui::Text("Here you can view statistics.");
			// Additional UI for statistical data
			ImGui::EndTabItem();
		}

		ImGui::EndTabBar();
	}

	ImGui::End(); // MainWindow

	if (ImGui::Begin("Debug"))
	{
		const auto allocated_blocks = g_allocated_blocks.load(std::memory_order_relaxed);
		const auto deallocated_blocks = g_deallocated_blocks.load(std::memory_order_relaxed);
		ImGui::Text("Allocated Blocks / Deallocated Blocks: (%ld / %ld) (%ld / %ld) (Total: %ld)", allocated_blocks, deallocated_blocks,
					g_image_list.images.size(), allocated_blocks - deallocated_blocks,
					g_image_list.images.size() + (allocated_blocks - deallocated_blocks));
	}
	ImGui::End();

	for (blt::size_t i = 0; i < population_size; i++)
		gl_images[i]->upload(get_image(i).data.data(), IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGBA, GL_FLOAT);

	constexpr int images_x = 10;
	constexpr int images_y = 6;
	for (int i = 0; i < images_x; i++)
	{
		for (int j = 0; j < images_y; j++)
		{
			constexpr float padding_x = 32;
			constexpr float padding_y = 32;
			const float img_width = (static_cast<float>(data.width) - padding_x * 2 - padding_x * (images_x-1) - 256) / images_x;
			const float img_height = (static_cast<float>(data.height) - padding_y * 2 - padding_y * (images_y-1) - 32) / images_y;
			const float x = 256 + static_cast<float>(i) * img_width + padding_x * static_cast<float>(i) + img_width;
			const float y = static_cast<float>(data.height) - (16 + static_cast<float>(j) * img_height + padding_y * static_cast<float>(j) +
				img_height);
			renderer_2d.drawRectangle(blt::gfx::rectangle2d_t{x, y, img_width, img_height}, std::to_string(i * j));
		}
	}

	renderer_2d.render(data.width, data.height);
}

void destroy(const blt::gfx::window_data&)
{
	gl_images.clear();
	global_matrices.cleanup();
	resources.cleanup();
	renderer_2d.cleanup();
	blt::gfx::cleanup();
}

int main()
{
	auto run_gp_thread = run_gp();
	blt::gfx::init(blt::gfx::window_data{"Image GP", init, update, destroy}.setSyncInterval(1));
	should_exit = true;
	run_gp_thread.join();
}
