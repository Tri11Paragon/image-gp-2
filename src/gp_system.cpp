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
#include "opencv2/imgproc.hpp"
#include <stb_perlin.h>

using namespace blt::gp;

float filter_nan(const float f)
{
	if (std::isnan(f) || std::isinf(f) || std::isinf(-f))
		return 0.0f;
	return f;
}

std::array<gp_program*, 3> programs;
prog_config_t config{};

std::vector<std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3>> images;
auto reference_image = image_storage_t::from_file("../silly.png");

template <size_t Channel>
void fitness_func(const tree_t& tree, fitness_t& fitness, const blt::size_t index)
{
	auto image = tree.get_evaluation_ref<image_t>();
	image->normalize();

	// std::memcpy(images[index].data.data(), image->get_data().data.data(), IMAGE_SIZE_BYTES);
	auto& data = image->get_data();
	for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
	{
		for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
		{
			images[index][(x * IMAGE_DIMENSIONS + y) * 3 + Channel] = data.get(x, y);

			auto multiplier = (1 - std::abs((static_cast<float>(x) / (static_cast<float>(IMAGE_DIMENSIONS) / 2)) - 1)) + (1 - std::abs(
				(static_cast<float>(y) / (static_cast<float>(IMAGE_DIMENSIONS) / 2)) - 1));
			const auto diff = filter_nan(data.get(x, y)) - reference_image[Channel].get(x, y);
			fitness.raw_fitness += diff * diff * multiplier;
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
		image_t ret{};
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				ret.get_data().get(x, y) = static_cast<float>(x) / static_cast<float>(IMAGE_DIMENSIONS - 1);
		}
		return ret;
	});
	static operation_t op_image_y([]() {
		image_t ret{};
		for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
				ret.get_data().get(x, y) = static_cast<float>(y) / static_cast<float>(IMAGE_DIMENSIONS - 1);
		}
		return ret;
	});
	static auto op_image_noise = operation_t([program]() {
		image_t ret{};
		for (auto& v : ret.get_data().data)
			v = program->get_random().get_float(0, 1.0f);
		return ret;
	}).set_ephemeral();
	static auto op_image_ephemeral = operation_t([program]() {
		image_t ret{};
		const auto value = program->get_random().get_float();
		for (auto& v : ret.get_data().data)
			v = value;
		return ret;
	}).set_ephemeral();
	static operation_t op_image_blend([](const image_t a, const image_t b, const float f) {
		const auto blend = std::min(std::max(f, 0.0f), 1.0f);
		const auto beta = 1.0f - blend;
		image_t ret{};
		const cv::Mat src1{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, a.as_void_const()};
		const cv::Mat src2{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, b.as_void_const()};
		cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, ret.get_data().data.data()};
		addWeighted(src1, blend, src2, beta, 0.0, dst);
		return ret;
	}, "blend_image");
	static operation_t op_image_sin([](const image_t a) {
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = filter_nan(std::sin(v));
		return ret;
	}, "sin_image");
	static operation_t op_image_cos([](const image_t a) {
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = filter_nan(std::cos(v));
		return ret;
	}, "cos_image");
	static operation_t op_image_log([](const image_t a) {
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
		{
			if (blt::f_equal(v, 0))
				ret.get_data().data[i] = 0;
			else if (v < 0)
				ret.get_data().data[i] = -filter_nan(std::log(-v));
			else
				ret.get_data().data[i] = filter_nan(std::log(v));
		}
		return ret;
	}, "log_image");
	static operation_t op_image_exp([](const image_t a) {
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = filter_nan(std::exp(v));
		return ret;
	}, "exp_image");
	static operation_t op_image_or([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = blt::mem::type_cast<float>(blt::mem::type_cast<blt::u32>(av) | blt::mem::type_cast<blt::u32>(bv));
		return ret;
	}, "bit_or_image");
	static operation_t op_image_and([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = blt::mem::type_cast<float>(blt::mem::type_cast<blt::u32>(av) & blt::mem::type_cast<blt::u32>(bv));
		return ret;
	}, "bit_and_image");
	static operation_t op_image_xor([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = blt::mem::type_cast<float>(blt::mem::type_cast<blt::u32>(av) ^ blt::mem::type_cast<blt::u32>(bv));
		return ret;
	}, "bit_xor_image");
	static operation_t op_image_not([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = blt::mem::type_cast<float>(~blt::mem::type_cast<blt::u32>(av));
		return ret;
	}, "bit_not_image");
	static operation_t op_image_gt([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = av > bv ? av : bv;
		return ret;
	}, "gt_image");
	static operation_t op_image_lt([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = av < bv ? av : bv;
		return ret;
	}, "lt_image");
	static operation_t op_image_perlin([](const float ofx, const float ofy, const float ofz, const float lacunarity, const float gain,
										const float octaves) {
		image_t ret{};
		for (const auto& [i, out] : blt::enumerate(ret.get_data().data))
			out = stb_perlin_ridge_noise3(static_cast<float>(i % IMAGE_DIMENSIONS) + ofx, static_cast<float>(i / IMAGE_DIMENSIONS) + ofy, ofz,
										lacunarity + 2, gain + 0.5f, 1.0f, static_cast<int>(octaves));
		return ret;
	}, "perlin_image");
	static operation_t op_image_perlin_bounded([](float octaves) {
		octaves = std::min(std::max(octaves, 4.0f), 8.0f);
		image_t ret{};
		for (const auto& [i, out] : blt::enumerate(ret.get_data().data))
			out = stb_perlin_fbm_noise3(static_cast<float>(i % IMAGE_DIMENSIONS) + 0.23423f, static_cast<float>(i / IMAGE_DIMENSIONS) + 0.6234f,
										0.4861f, 2, 0.5, static_cast<int>(octaves));
		return ret;
	}, "perlin_image_bounded");
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
	static operation_t op_erode([](const image_t a, float erosion_size) {
		image_t ret{};
		erosion_size = std::min(std::max(erosion_size, 0.0f), 21.0f);
		const cv::Mat src{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, a.as_void_const()};
		cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, ret.get_data().data.data()};
		const cv::Mat element = cv::getStructuringElement( cv::MORPH_CROSS,
							 cv::Size( static_cast<int>(2*erosion_size + 1), static_cast<int>(2*erosion_size+1) ),
							 cv::Point( static_cast<int>(erosion_size), static_cast<int>(erosion_size) ) );
		cv::erode( src, dst, element );
		return ret;
	}, "erode_image");

	static operation_t op_dilate([](const image_t a, float dilate_size) {
		image_t ret{};
		dilate_size = std::min(std::max(dilate_size, 0.0f), 21.0f);
		const cv::Mat src{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, a.as_void_const()};
		cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, ret.get_data().data.data()};
		const cv::Mat element = cv::getStructuringElement( cv::MORPH_CROSS,
							 cv::Size( static_cast<int>(2*dilate_size + 1), static_cast<int>(2*dilate_size+1) ),
							 cv::Point( static_cast<int>(dilate_size), static_cast<int>(dilate_size) ) );
		cv::dilate( src, dst, element );
		return ret;
	}, "erode_image");

	operator_builder builder{};
	builder.build(op_image_ephemeral, make_add<image_t>(), make_sub<image_t>(), make_mul<image_t>(), make_div<image_t>(), op_image_x, op_image_y,
				op_image_sin, op_image_gt, op_image_lt, op_image_cos, op_image_log, op_image_exp, op_image_or, op_image_and, op_image_xor,
				op_image_not, op_image_perlin_bounded, op_image_blend, op_erode, op_dilate, make_add<float>(), make_sub<float>(), make_mul<float>(),
				make_prot_div<float>(), op_sin, op_cos, op_exp, op_log, lit);
	program->set_operations(builder.grab());
}

void setup_gp_system(const blt::size_t population_size)
{
	config.set_pop_size(population_size);
	config.set_elite_count(2);
	config.set_thread_count(0);
	config.set_reproduction_chance(0);
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

std::array<image_storage_t, 3>& get_reference_image()
{
	return reference_image;
}

std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3> to_gl_image(const std::array<image_storage_t, 3>& image)
{
	std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3> image_data{};
	for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
	{
		for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
		{
			image_data[(x * IMAGE_DIMENSIONS + y) * 3 + 0] = image[0].get(x, y);
			image_data[(x * IMAGE_DIMENSIONS + y) * 3 + 1] = image[1].get(x, y);
			image_data[(x * IMAGE_DIMENSIONS + y) * 3 + 2] = image[2].get(x, y);
		}
	}
	return image_data;
}

blt::u32 get_generation()
{
	return programs[0]->get_current_generation();
}

void set_population_size(const blt::u32 size)
{
	if (size > images.size())
		images.resize(size);
	config.set_pop_size(size);
	for (const auto program : programs)
		program->set_config(config);
	reset_programs();
}

void reset_programs()
{
	for (const auto program : programs)
		program->reset_program(program->get_typesystem().get_type<image_t>().id());
}

std::array<size_t, 3> get_best_image_index()
{
	std::array<size_t, 3> best_index{};
	for (const auto [slot, program] : blt::zip(best_index, programs))
		slot = program->get_best_indexes<1>()[0];
	return best_index;
}

void regenerate_image(blt::size_t index, float& image_storage, blt::i32 width, blt::i32 height)
{

}
