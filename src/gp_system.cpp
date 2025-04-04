/*
 *  <Short Description>
 *  Copyright (C) 2025  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <gp_system.h>
#include <blt/gp/program.h>
#include <image_storage.h>
#include <operations.h>
#include <random>
#include "opencv2/imgcodecs.hpp"

using namespace blt::gp;

std::array<gp_program*, 3> programs;

std::vector<std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3>> images;
auto reference_image = image_storage_t::from_file("../silly.png");

template <size_t Channel>
void fitness_func(const tree_t& tree, fitness_t& fitness, const blt::size_t index)
{
	auto image = tree.get_evaluation_ref<image_t>();

	// std::memcpy(images[index].data.data(), image->get_data().data.data(), IMAGE_SIZE_BYTES);
	auto& data = image->get_data();
	for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
	{
		for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
		{
			images[index][(x * IMAGE_DIMENSIONS + y) * 3 + Channel] = data.get(x, y);

			auto multiplier = (1 - std::abs((static_cast<float>(x) / (static_cast<float>(IMAGE_DIMENSIONS) / 2)) - 1)) + (1 - std::abs(
				(static_cast<float>(y) / (static_cast<float>(IMAGE_DIMENSIONS) / 2)) - 1));
			const auto diff = data.get(x, y) - reference_image[Channel].get(x, y);
			fitness.raw_fitness += (diff * diff) * multiplier;
		}
	}
	fitness.raw_fitness /= static_cast<float>(IMAGE_SIZE_CHANNELS);
	fitness.standardized_fitness = fitness.raw_fitness;
	fitness.adjusted_fitness = -fitness.standardized_fitness;
}

template <typename T>
void setup_operations(gp_program* program)
{
	static operation_t op_image_x([]() {
		image_t ret;
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				ret.get_data().get(x, y) = static_cast<float>(x) / static_cast<float>(IMAGE_DIMENSIONS - 1);
		}
		return ret;
	});
	static operation_t op_image_y([]() {
		image_t ret;
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				ret.get_data().get(x, y) = static_cast<float>(y) / static_cast<float>(IMAGE_DIMENSIONS - 1);
		}
		return ret;
	});
	static auto op_image_ephemeral = operation_t([program]() {
		image_t ret;
		const auto value = program->get_random().get_float();
		for (auto& v : ret.get_data().data)
			v = value;
		return ret;
	}).set_ephemeral();
	static operation_t op_image_blend([](const image_t& a, const image_t& b, const float f) {
		const auto blend = std::min(std::max(f, 0.0f), 1.0f);
		const auto beta = 1.0f - blend;
		image_t ret;
		const cv::Mat src1{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32FC1, a.as_void()};
		const cv::Mat src2{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32FC1, b.as_void()};
		cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32FC1, ret.get_data().data.data()};
		addWeighted(src1, blend, src2, beta, 0.0, dst);
		return ret;
	}, "blend_image");
	static operation_t op_image_sin([](const image_t& a) {
		image_t ret;
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = std::sin(v);
		return ret;
	}, "sin_image");
	static operation_t op_image_cos([](const image_t& a) {
		image_t ret;
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = std::cos(v);
		return ret;
	}, "cos_image");
	static operation_t op_image_log([](const image_t& a) {
		image_t ret;
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
		{
			if (blt::f_equal(v, 0))
				ret.get_data().data[i] = 0;
			else if (v < 0)
				ret.get_data().data[i] = -std::log(-v);
			else
				ret.get_data().data[i] = std::log(v);
		}
		return ret;
	}, "sin_image");
	static operation_t op_sin([](const float a) {
		return std::sin(a);
	}, "sin_float");
	static operation_t op_cos([](const float a) {
		return std::cos(a);
	}, "cos_float");
	static operation_t op_exp([](const float a) {
		return std::exp(a);
	}, "exp_float");
	static operation_t op_log([](const float a) {
		if (blt::f_equal(a, 0))
			return 0.0f;
		if (a < 0)
			return -std::log(-a);
		return std::log(a);
	}, "log_float");
	static auto lit = operation_t([program]() {
		return program->get_random().get_float(-1.0f, 1.0f);
	}, "lit_float").set_ephemeral();

	operator_builder builder{};
	builder.build(make_add<image_t>(), make_sub<image_t>(), make_mul<image_t>(), make_div<image_t>(), op_image_x, op_image_y, op_image_sin,
				 op_image_cos, op_image_log, make_add<float>(), make_sub<float>(), make_mul<float>(), make_prot_div<float>(),
				op_sin, op_cos, op_exp, op_log, lit);
	program->set_operations(builder.grab());
}

void setup_gp_system(const blt::size_t population_size)
{
	prog_config_t config{};
	config.set_pop_size(1);
	config.set_elite_count(0);
	config.set_thread_count(0);
	// config.set_crossover_chance(0);
	// config.set_mutation_chance(0);
	// config.set_reproduction_chance(0);

	const auto rand = std::random_device()();
	BLT_INFO("Random Seed: {}", rand);
	for (auto& program : programs)
	{
		program = new gp_program{rand, config};
	}
	setup_operations<struct p1>(programs[0]);
	setup_operations<struct p2>(programs[1]);
	setup_operations<struct p3>(programs[2]);

	images.resize(population_size);

	static auto sel = select_tournament_t{};

	for (const auto program : programs)
		program->generate_initial_population(program->get_typesystem().get_type<image_t>().id());

	programs[0]->setup_generational_evaluation(fitness_func<0>, sel, sel, sel);
	programs[1]->setup_generational_evaluation(fitness_func<1>, sel, sel, sel);
	programs[2]->setup_generational_evaluation(fitness_func<2>, sel, sel, sel);
}

void run_step()
{
	BLT_TRACE("------------\\{Begin Generation {}}------------", programs[0]->get_current_generation());
	BLT_TRACE("Creating next generation");
	for (const auto program : programs)
		program->create_next_generation();
	BLT_TRACE("Move to next generation");
	for (const auto program : programs)
		program->next_generation();
	BLT_TRACE("Evaluate Fitness");
	for (const auto program : programs)
		program->evaluate_fitness();
	for (const auto [i, program] : blt::enumerate(programs))
	{
		const auto& stats = program->get_population_stats();
		if (i == 0)
			BLT_TRACE("Channel Red");
		else if (i == 1)
			BLT_TRACE("Channel Green");
		else
			BLT_TRACE("Channel Blue");
		BLT_TRACE("\tAvg Fit: {:0.6f}, Best Fit: {:0.6f}, Worst Fit: {:0.6f}, Overall Fit: {:0.6f}",
				stats.average_fitness.load(std::memory_order_relaxed), stats.best_fitness.load(std::memory_order_relaxed),
				stats.worst_fitness.load(std::memory_order_relaxed), stats.overall_fitness.load(std::memory_order_relaxed));
	}

	BLT_TRACE("----------------------------------------------");
}

bool should_terminate()
{
	return programs[0]->should_terminate() || programs[1]->should_terminate() || programs[2]->should_terminate();
}

std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3>& get_image(const blt::size_t index)
{
	return images[index];
}

void cleanup()
{
	for (const auto program : programs)
		delete program;
}
