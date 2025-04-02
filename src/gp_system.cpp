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

gp_program program{
	[]() {
		return std::random_device()();
	}
};

image_storage_t reference_image = image_storage_t::from_file("../silly.png");

void fitness_func(const tree_t& tree, fitness_t& fitness, const blt::size_t)
{
	auto image = tree.get_evaluation_ref<image_t>();
	auto& data = image->get_data();
	for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
	{
		for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
		{
			auto multiplier = (1 - std::abs((static_cast<float>(x) / (static_cast<float>(IMAGE_DIMENSIONS) / 2)) - 1)) + (1 - std::abs(
				(static_cast<float>(y) / (static_cast<float>(IMAGE_DIMENSIONS) / 2)) - 1));
			for (blt::size_t c = 0; c < IMAGE_CHANNELS; ++c)
			{
				const auto diff_r = data.get(x, y, 0) - reference_image.get(x, y, 0);
				const auto diff_g = data.get(x, y, 1) - reference_image.get(x, y, 1);
				const auto diff_b = data.get(x, y, 2) - reference_image.get(x, y, 2);
				fitness.raw_fitness += diff_r * diff_r * multiplier + diff_g * diff_g * multiplier + diff_b * diff_b * multiplier;
			}
		}
	}
	fitness.raw_fitness /= static_cast<float>(IMAGE_DIMENSIONS * IMAGE_DIMENSIONS);
	fitness.standardized_fitness = fitness.raw_fitness;
	fitness.adjusted_fitness = fitness.standardized_fitness;
}

void setup_operations()
{
	static operation_t op_image_x([]() {
		image_t ret;
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				for (blt::i32 c = 0; c < IMAGE_CHANNELS; ++c)
					ret.get_data().get(x, y, c) = static_cast<float>(x) / static_cast<float>(IMAGE_DIMENSIONS - 1);
		}
		return ret;
	});
	static operation_t op_image_y([]() {
		image_t ret;
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				for (blt::i32 c = 0; c < IMAGE_CHANNELS; ++c)
					ret.get_data().get(x, y, c) = static_cast<float>(y) / static_cast<float>(IMAGE_DIMENSIONS - 1);
		}
		return ret;
	});
	static operation_t op_image_xy([]() {
		image_t ret;
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				for (blt::i32 c = 0; c < IMAGE_CHANNELS; ++c)
					ret.get_data().get(x, y, c) = static_cast<float>(x + y) / static_cast<float>((IMAGE_DIMENSIONS - 1) * (IMAGE_DIMENSIONS - 1));
		}
		return ret;
	});
	static operation_t op_image_blend([](const image_t& a, const image_t& b, const float f) {
		const auto blend = std::min(std::max(f, 0.0f), 1.0f);
		const auto beta = 1.0f - blend;
		image_t ret;
		const cv::Mat src1{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32FC4, a.as_void()};
		const cv::Mat src2{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32FC4, b.as_void()};
		cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32FC4, ret.get_data().data.data()};
		addWeighted(src1, blend, src2, beta, 0.0, dst);
		return ret;
	}, "blend");
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
		return a <= 0.0f ? 0.0f : std::log(a);
	}, "log_float");
	static auto lit = operation_t([]() {
		return program.get_random().get_float(-1.0f, 1.0f);
	}, "lit_float").set_ephemeral();

	operator_builder builder{};
	builder.build(make_add<image_t>(), make_sub<image_t>(), make_mul<image_t>(), make_div<image_t>(), op_image_x, op_image_y, op_image_xy,
				make_add<float>(), make_sub<float>(), make_mul<float>(), make_prot_div<float>(), op_sin, op_cos, op_exp, op_log, lit);
	program.set_operations(builder.grab());
}

void setup_gp_system(const blt::size_t population_size)
{
	prog_config_t config{};
	config.population_size = population_size;
	program.set_config(config);
	setup_operations();
	static auto sel = select_tournament_t{};
	program.generate_initial_population(program.get_typesystem().get_type<image_t>().id());
	program.setup_generational_evaluation(fitness_func, sel, sel, sel);
}

void run_step()
{
	BLT_TRACE("------------\\{Begin Generation {}}------------", program.get_current_generation());
	BLT_TRACE("Creating next generation");
	program.create_next_generation();
	BLT_TRACE("Move to next generation");
	program.next_generation();
	BLT_TRACE("Evaluate Fitness");
	program.evaluate_fitness();
	const auto& stats = program.get_population_stats();
	BLT_TRACE("Avg Fit: {:0.6f}, Best Fit: {:0.6f}, Worst Fit: {:0.6f}, Overall Fit: {:0.6f}", stats.average_fitness.load(std::memory_order_relaxed),
			stats.best_fitness.load(std::memory_order_relaxed), stats.worst_fitness.load(std::memory_order_relaxed),
			stats.overall_fitness.load(std::memory_order_relaxed));
	BLT_TRACE("----------------------------------------------");
}

bool should_terminate()
{
	return program.should_terminate();
}
