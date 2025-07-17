#include <gp_system.h>

#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/camera.h"
#include <imgui.h>
#include <thread>
#include <implot.h>
#include <blt/gp/tree.h>

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::batch_renderer_2d renderer_2d(resources, global_matrices);
blt::gfx::first_person_camera camera;

std::vector<blt::gfx::texture_gl2D*> gl_images;
blt::size_t population_size = 64;

namespace im = ImGui;

std::atomic_bool run_generation = false;
std::atomic_bool should_exit = false;

void update_population_size(const blt::u32 new_size)
{
	using namespace blt::gfx;
	if (new_size == population_size)
		return;
	for (blt::size_t i = population_size; i < new_size; i++)
	{
		auto texture = new texture_gl2D(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGBA8);
		texture->bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		gl_images.push_back(texture);
		resources.set(std::to_string(i), texture);
	}
	set_population_size(new_size);
	population_size = new_size;
}

std::thread run_gp()
{
	return std::thread{
		[]() {
			while (!should_exit)
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
	ImPlot::CreateContext();
	using namespace blt::gfx;

	for (blt::size_t i = 0; i < population_size; i++)
	{
		auto texture = new texture_gl2D(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGBA8);
		texture->bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		gl_images.push_back(texture);
		resources.set(std::to_string(i), texture);
	}
	const auto texture = new texture_gl2D(IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGBA8);
	texture->bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	texture->upload(to_gl_image(get_reference_image()).data(), IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGB, GL_FLOAT);
	resources.set("reference", texture);
	global_matrices.create_internals();
	resources.load_resources();
	renderer_2d.create();

	setWindowSize(1400, 720);
}

void update(const blt::gfx::window_data& data)
{
	static float side_bar_width = 260;
	static float top_bar_height = 32;

	global_matrices.update_perspectives(data.width, data.height, 90, 0.1, 2000);

	camera.update();
	camera.update_view(global_matrices);
	global_matrices.update();

	im::ShowDemoWindow();

	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y));
	ImGui::Begin("MainWindow", nullptr,
				ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
				ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground);

	static blt::i32 image_to_enlarge = -1;
	bool clicked_on_image = false;
	static bool use_gramma_correction = false;
	static bool show_best = false;

	// Create the tab bar
	if (ImGui::BeginTabBar("MainTabs"))
	{
		constexpr float padding_x = 16;
		constexpr float padding_y = 16;
		// 1. Run GP tab
		static int images_x = 10;
		static int images_y = 6;
		static bool run_gp = false;
		static int generation_limit = 0;
		static int min_between_runs = 100;

		if (ImGui::BeginTabItem("Run GP"))
		{
			ImGui::BeginChild("ControlPanel", ImVec2(250, 0), true);
			{
				ImGui::Text("Control Panel");
				ImGui::Separator();
				if (ImGui::Button("Run Step"))
					run_generation = true;
				ImGui::Checkbox("Run GP", &run_gp);
				if (ImGui::InputInt("Images X", &images_x) || ImGui::InputInt("Images Y", &images_y))
					update_population_size(images_x * images_y);
				ImGui::InputInt("Generation Limit", &generation_limit);
				if (run_gp && (generation_limit == 0 || get_generation() < static_cast<blt::u32>(generation_limit)))
					run_generation = true;
				ImGui::InputInt("Min Time Between Runs (ms)", &min_between_runs);
				ImGui::Checkbox("Show Best", &show_best);
				if (ImGui::Checkbox("Use Gamma Correction?", &use_gramma_correction))
					set_use_gamma_correction(use_gramma_correction);
			}
			ImGui::EndChild();

			// Right child - take the remaining space
			ImGui::SameLine();
			// ImGui::BeginChild("MainContent", ImVec2(0, 0), false, ImGuiWindowFlags_NoBackground);
			{
				const auto area_x = static_cast<float>(data.width) - side_bar_width;
				const auto area_y = static_cast<float>(data.height) - top_bar_height;

				const auto padding_area_x = (static_cast<float>(images_x) + 2) * padding_x;
				const auto padding_area_y = (static_cast<float>(images_y) + 2) * padding_y;

				const auto area_width = area_x - padding_area_x;
				const auto area_height = area_y - padding_area_y;

				const auto image_width = area_width / static_cast<float>(images_x);
				const auto image_height = area_height / static_cast<float>(images_y);

				for (int i = 0; i < images_x; i++)
				{
					for (int j = 0; j < images_y; j++)
					{
						int image_at_pos = i * images_y + j;
						const auto mx = static_cast<float>(blt::gfx::getMouseX());
						const auto my = static_cast<float>(data.height) - static_cast<float>(blt::gfx::getMouseY());
						const auto x = side_bar_width + static_cast<float>(i) * (image_width + padding_x) + image_width / 2 + padding_x;
						const auto y = static_cast<float>(j) * (image_height + padding_y) + image_height / 2 + padding_y / 2;

						blt::vec2 extra;
						if (mx >= x - image_width / 2 && mx <= x + image_width / 2 && my >= y - image_height / 2 && my <= y + image_height / 2)
						{
							extra = {padding_x / 2.0f, padding_y / 2.0f};
							if (blt::gfx::isMousePressed(0) && blt::gfx::mousePressedLastFrame())
							{
								image_to_enlarge = image_at_pos;
								clicked_on_image = true;
							}
						}

						renderer_2d.drawRectangle(blt::gfx::rectangle2d_t{x, y, image_width + extra.x(), image_height + extra.y()},
												std::to_string(image_at_pos));
					}
				}
			}
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
			auto [mean_chan, variance_chan] = get_mean_and_variance();

			for (const auto [i, mean] : blt::enumerate(mean_chan))
			{
				std::string type;
				switch (i)
				{
					case 0:
						type = "Red";
						break;
					case 1:
						type = "Green";
						break;
					case 2:
						type = "Blue";
						break;
					default: break;
				}

				if (ImPlot::BeginPlot(("Mean Graph " + type).c_str()))
				{
					ImPlot::PlotLine("Mean", mean.data(), static_cast<int>(mean.size()));
					ImPlot::EndPlot();
				}
			}

			for (const auto [i, variance] : blt::enumerate(variance_chan))
			{
				std::string type;
				switch (i)
				{
					case 0:
						type = "Red";
						break;
					case 1:
						type = "Green";
						break;
					case 2:
						type = "Blue";
						break;
					default: break;
				}
				if (ImPlot::BeginPlot(("Variance Graph " + type).c_str()))
				{
					ImPlot::PlotLine("Variance", variance.data(), static_cast<int>(variance.size()));
					ImPlot::EndPlot();
				}
			}

			auto pops = get_populations();

			const std::array<std::string, 3> labels = {"Red", "Green", "Blue"};

			for (const auto& [i, label, pop] : blt::in_pairs(labels, pops).enumerate().flatten())
			{
				if (i > 0)
					ImGui::SameLine();
				ImGui::BeginGroup();
				ImGui::Text("Population (%s)", label.c_str());
				if (ImGui::BeginChild(label.c_str(), ImVec2(250, 0), true))
				{
					for (const auto& [i, ind] : blt::enumerate(*pop))
					{
						ImGui::Text("Tree (%ld) -> Fitness: %lf", i, ind.fitness.adjusted_fitness);
					}
				}
				ImGui::EndChild();
				ImGui::EndGroup();
			}

			// Additional UI for statistical data
			ImGui::EndTabItem();
		}

		if (ImGui::BeginTabItem("Reference"))
		{
			auto w = static_cast<float>(data.width);
			auto h = static_cast<float>(data.height) - top_bar_height - 10;

			renderer_2d.drawRectangle({
										w / 2,
										h / 2,
										std::min(w, static_cast<float>(IMAGE_DIMENSIONS)),
										std::min(h, static_cast<float>(IMAGE_DIMENSIONS))
									}, "reference");
			ImGui::EndTabItem();
		}

		ImGui::EndTabBar();
	}

	ImGui::End(); // MainWindow

	// if (ImGui::Begin("Fitness"))
	// {
	// 	const auto& [average, best, worst, overall] = get_fitness_history();
	//
	// 	if (ImPlot::BeginPlot("Average Fitness", ImVec2{-1, 0}, ImPlotFlags_NoInputs))
	// 	{
	// 		ImPlot::PlotLine("Average Fitness", average.data(), static_cast<blt::i32>(average.size()));
	// 		ImPlot::EndPlot();
	// 	}
	// 	if (ImPlot::BeginPlot("Best Fitness", ImVec2{-1, 0}, ImPlotFlags_NoInputs))
	// 	{
	// 		ImPlot::PlotLine("Best Fitness", best.data(), static_cast<blt::i32>(best.size()));
	// 		ImPlot::EndPlot();
	// 	}
	// 	if (ImPlot::BeginPlot("Worst Fitness", ImVec2{-1, 0}, ImPlotFlags_NoInputs))
	// 	{
	// 		ImPlot::PlotLine("Worst Fitness", worst.data(), static_cast<blt::i32>(worst.size()));
	// 		ImPlot::EndPlot();
	// 	}
	// 	if (ImPlot::BeginPlot("Overall Fitness", ImVec2{-1, 0}, ImPlotFlags_NoInputs))
	// 	{
	// 		ImPlot::PlotLine("Overall Fitness", overall.data(), static_cast<blt::i32>(overall.size()));
	// 		ImPlot::EndPlot();
	// 	}
	// }
	// ImGui::End();

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
	{
		gl_images[i]->upload(get_image(i).data(), IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, GL_RGB, GL_UNSIGNED_INT);
	}

	if ((blt::gfx::isMousePressed(0) && blt::gfx::mousePressedLastFrame() && !clicked_on_image) || (blt::gfx::isKeyPressed(GLFW_KEY_ESCAPE) &&
		blt::gfx::keyPressedLastFrame()))
		image_to_enlarge = -1;

	if (show_best)
	{
		auto best_images = get_best_image_index();
		for (const auto [i, best_image] : blt::enumerate(best_images))
		{
			const auto width = std::min(static_cast<float>(data.width) - side_bar_width, static_cast<float>(256) * 3) / 3;
			const auto height = std::min(static_cast<float>(data.height) - top_bar_height, static_cast<float>(256) * 3) / 3;
			renderer_2d.drawRectangle(blt::gfx::rectangle2d_t{blt::gfx::anchor_t::BOTTOM_LEFT, side_bar_width + 256 + i * width, 64, width, height},
									std::to_string(best_image), 1);
		}
	}

	if (image_to_enlarge != -1)
	{
		if (blt::gfx::isKeyPressed(GLFW_KEY_R) && blt::gfx::keyPressedLastFrame())
		{}
		renderer_2d.drawRectangle(blt::gfx::rectangle2d_t{
									blt::gfx::anchor_t::BOTTOM_LEFT,
									side_bar_width + 256,
									64,
									std::min(static_cast<float>(data.width) - side_bar_width, static_cast<float>(256) * 3),
									std::min(static_cast<float>(data.height) - top_bar_height, static_cast<float>(256) * 3)
								}, std::to_string(image_to_enlarge), 1);
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
	setup_gp_system(population_size);
	auto run_gp_thread = run_gp();
	blt::gfx::init(blt::gfx::window_data{"Image GP", init, update, destroy}.setSyncInterval(1));
	should_exit = true;
	run_gp_thread.join();
}
