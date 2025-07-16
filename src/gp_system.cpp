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

bool is_nan(const float f)
{
	return std::isnan(f) || std::isinf(f) || std::isinf(-f);
}

float filter_nan(const float f, const float failure = 0.0f)
{
	if (is_nan(f))
		return failure;
	return f;
}

bool use_gamma_correction = false;

std::array<gp_program*, 3> programs;
prog_config_t config{};

std::vector<std::array<image_ipixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3>> images;
std::vector<std::array<image_ipixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS>> images_red;
std::vector<std::array<image_ipixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS>> images_green;
std::vector<std::array<image_ipixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS>> images_blue;
std::array<image_storage_t, 3> reference_image;

std::vector<float> average_fitness;
std::vector<float> best_fitness;
std::vector<float> worst_fitness;
std::vector<float> overall_fitness;

template <size_t Channel>
void fitness_func(const tree_t& tree, fitness_t& fitness, const blt::size_t index)
{
	auto image = tree.get_evaluation_ref<image_t>();

	auto& data = image->get_data();
	for (blt::size_t x = 0; x < IMAGE_DIMENSIONS; ++x)
	{
		for (blt::size_t y = 0; y < IMAGE_DIMENSIONS; ++y)
		{
			switch (Channel)
			{
				case 0:
					images_red[index][(y * IMAGE_DIMENSIONS + x)] = data.get(x, y);
					break;
				case 1:
					images_green[index][(y * IMAGE_DIMENSIONS + x)] = data.get(x, y);
					break;
				case 2:
					images_blue[index][(y * IMAGE_DIMENSIONS + x)] = data.get(x, y);
					break;
				default:
					break;
			}
			// images[index][(y * IMAGE_DIMENSIONS + x) * 3 + Channel] = data.get(x, y);

			const auto our = static_cast<double>(data.get(x, y)) / static_cast<double>(std::numeric_limits<blt::u32>::max());

			const auto theirs = reference_image[Channel].get(x, y);

			const auto gamma_ours = std::pow(our, 1.0f / 2.2f);
			// const auto gamma_theirs = std::pow(theirs, 1.0f / 2.2f);

			if (use_gamma_correction)
			{
				const auto diff = gamma_ours - theirs;
				fitness.raw_fitness += static_cast<float>((diff * diff));
			} else
			{
				const auto diff = our - theirs;
				fitness.raw_fitness += static_cast<float>((diff * diff));
			}
		}
	}
	fitness.set_normal(static_cast<float>(std::sqrt(fitness.raw_fitness)));
	// fitness.raw_fitness = static_cast<float>(std::sqrt(fitness.raw_fitness));
	// fitness.standardized_fitness = fitness.raw_fitness;
	// fitness.adjusted_fitness = -fitness.standardized_fitness;
}

template <typename T>
void setup_operations(gp_program* program)
{
	static operation_t op_image_x([]() {
		constexpr auto mul = std::numeric_limits<blt::u32>::max() / static_cast<blt::u32>(IMAGE_DIMENSIONS - 1);
		image_t ret{};
		for (blt::u32 x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::u32 y = 0; y < IMAGE_DIMENSIONS; ++y)
				ret.get_data().get(x, y) = y * mul;
		}
		return ret;
	});
	static operation_t op_image_y([]() {
		constexpr auto mul = std::numeric_limits<blt::u32>::max() / static_cast<blt::u32>(IMAGE_DIMENSIONS - 1);
		image_t ret{};
		for (blt::u32 x = 0; x < IMAGE_DIMENSIONS; ++x)
		{
			for (blt::u32 y = 0; y < IMAGE_DIMENSIONS; ++y)
				ret.get_data().get(x, y) = x * mul;
		}
		return ret;
	});
	static auto op_image_random = operation_t([program]() {
		image_t ret{};
		for (auto& v : ret.get_data().data)
			v = program->get_random().get_u32(0, std::numeric_limits<blt::u32>::max());
		return ret;
	});
	static auto op_image_noise = operation_t([program]() {
		image_t ret{};
		for (auto& v : ret.get_data().data)
			v = program->get_random().get_u32(0, std::numeric_limits<blt::u32>::max());
		return ret;
	}).set_ephemeral();
	static auto op_image_ephemeral = operation_t([program]() {
		image_t ret{};
		const auto value = program->get_random().get_u32(0, std::numeric_limits<blt::u32>::max());
		for (auto& v : ret.get_data().data)
			v = value;
		return ret;
	}).set_ephemeral();
	// static operation_t op_image_blend([](const image_t a, const image_t b, const float f) {
	// 	const auto blend = std::min(std::max(f, 0.0f), 1.0f);
	// 	const auto beta = 1.0f - blend;
	// 	image_t ret{};
	// 	const cv::Mat src1{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, a.as_void_const()};
	// 	const cv::Mat src2{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, b.as_void_const()};
	// 	cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, ret.get_data().data.data()};
	// 	addWeighted(src1, blend, src2, beta, 0.0, dst);
	// 	return ret;
	// }, "blend_image");
	static operation_t op_image_sin([](const image_t a) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = static_cast<blt::u32>(((std::sin((v / limit) * blt::PI) + 1.0) / 2.0f) * limit);
		return ret;
	}, "sin_image");
	static operation_t op_image_sin_off([](const image_t a, const image_t b) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, v, off] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = static_cast<blt::u32>(((std::sin((v / limit) * blt::PI * (off / (limit / 4))) + 1.0) / 2.0f) * limit);
		return ret;
	}, "sin_image_off");
	static operation_t op_image_cos([](const image_t a) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = static_cast<blt::u32>(((std::cos((v / limit) * blt::PI * 2) + 1.0) / 2.0f) * limit);
		return ret;
	}, "cos_image");
	static operation_t op_image_cos_off([](const image_t a, const image_t b) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, v, off] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = static_cast<blt::u32>(((std::cos((v / limit) * blt::PI * (off / (limit / 2))) + 1.0) / 2.0f) * limit);
		return ret;
	}, "cos_image_off");
	static operation_t op_image_log([](const image_t a) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
		{
			if (v == 0)
				ret.get_data().data[i] = 0;
			else
				ret.get_data().data[i] = static_cast<blt::u32>(std::log(v / limit) * limit);
		}
		return ret;
	}, "log_image");
	static operation_t op_image_exp([](const image_t a) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, v] : blt::enumerate(std::as_const(a.get_data().data)))
			ret.get_data().data[i] = static_cast<blt::u32>(std::exp(v / limit) * limit);
		return ret;
	}, "exp_image");
	static operation_t op_image_or([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = av | bv;
		return ret;
	}, "bit_or_image");
	static operation_t op_image_and([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = av & bv;
		return ret;
	}, "bit_and_image");
	static operation_t op_image_xor([](const image_t a, const image_t b) {
		image_t ret{};
		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
			ret.get_data().data[i] = av ^ bv;
		return ret;
	}, "bit_xor_image");
	static operation_t op_image_not([](const image_t a) {
		image_t ret{};
		for (const auto& [i, av] : blt::enumerate(std::as_const(a.get_data().data)).flatten())
			ret.get_data().data[i] = ~av;
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
	static operation_t op_image_grad([](const image_t a, const image_t b) {
		image_t out{};

		for (const auto& [i, av, bv] : blt::in_pairs(std::as_const(a.get_data().data), std::as_const(b.get_data().data)).enumerate().flatten())
		{
			const auto p = static_cast<double>(i) / static_cast<double>(IMAGE_SIZE);
			const auto pi = 1 - p;

			out.get_data().data[i] = static_cast<blt::u32>(av * p + bv * pi);
		}
		return out;
	}, "grad_image");
	static operation_t op_image_perlin([](const image_t a) {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		for (const auto& [i, out, bv] : blt::in_pairs(ret.get_data().data, std::as_const(a.get_data().data)).enumerate().flatten())
		{
			constexpr auto AND = IMAGE_DIMENSIONS - 1;
			const double y = (static_cast<float>(i) / IMAGE_DIMENSIONS) / static_cast<float>(IMAGE_DIMENSIONS);
			const double x = static_cast<float>(i & AND) / static_cast<float>(IMAGE_DIMENSIONS);
			out = static_cast<blt::u32>(stb_perlin_noise3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(bv / (limit * 0.1)), 0, 0,
														0) * limit);
		}
		return ret;
	}, "perlin_image");
	static auto op_image_2d_perlin_eph = operation_t([program]() {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		const auto variety = program->get_random().get_float(1.5, 255);
		const auto x_warp = program->get_random().get_i32(0, 255);
		const auto y_warp = program->get_random().get_i32(0, 255);
		const auto z_warp = program->get_random().get_i32(0, 255);

		const auto offset_x = program->get_random().get_float(1.0 / 64.0f, 16.0f);
		const auto offset_y = program->get_random().get_float(1.0 / 64.0f, 16.0f);

		for (const auto& [i, out] : blt::enumerate(ret.get_data().data))
		{
			constexpr auto AND = IMAGE_DIMENSIONS - 1;
			const double y = (static_cast<float>(i) / IMAGE_DIMENSIONS) / static_cast<float>(IMAGE_DIMENSIONS);
			const double x = static_cast<float>(i & AND) / static_cast<float>(IMAGE_DIMENSIONS);
			out = static_cast<blt::u32>(stb_perlin_noise3(static_cast<float>(x) * offset_x, static_cast<float>(y) * offset_y, variety, x_warp, y_warp,
														z_warp) * limit);
		}
		return ret;
	}, "perlin_image_eph").set_ephemeral();
	static auto op_image_2d_perlin_oct = operation_t([program]() {
		constexpr auto limit = static_cast<double>(std::numeric_limits<blt::u32>::max());
		image_t ret{};
		const auto rand = program->get_random().get_float(0, 255);
		const auto octaves = program->get_random().get_i32(2, 8);
		const auto gain = program->get_random().get_float(0.1f, 0.9f);
		const auto lac = program->get_random().get_float(1.5f, 6.f);

		const auto offset = program->get_random().get_float(1.0 / 255.0f, 16.0f);

		for (const auto& [i, out] : blt::enumerate(ret.get_data().data))
		{
			constexpr auto AND = IMAGE_DIMENSIONS - 1;
			const double y = (static_cast<float>(i) / IMAGE_DIMENSIONS) / static_cast<float>(IMAGE_DIMENSIONS);
			const double x = static_cast<float>(i & AND) / static_cast<float>(IMAGE_DIMENSIONS);
			out = static_cast<blt::u32>(stb_perlin_fbm_noise3(static_cast<float>(x * offset), static_cast<float>(y * offset), rand, lac, gain,
															octaves) * limit);
		}
		return ret;
	}, "perlin_image_eph_oct").set_ephemeral();

	static operation_t op_passthrough([](const image_t& a) {
		image_t ret{};
		std::memcpy(ret.get_data().data.data(), a.get_data().data.data(), IMAGE_SIZE_BYTES);
		return ret;
	}, "passthrough");

	// static operation_t op_erode([](const image_t a, float erosion_size) {
	// 	image_t ret{};
	// 	erosion_size = std::min(std::max(erosion_size, 0.0f), 21.0f);
	// 	const cv::Mat src{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, a.as_void_const()};
	// 	cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, ret.get_data().data.data()};
	// 	const cv::Mat element = cv::getStructuringElement( cv::MORPH_CROSS,
	// 						 cv::Size( static_cast<int>(2*erosion_size + 1), static_cast<int>(2*erosion_size+1) ),
	// 						 cv::Point( static_cast<int>(erosion_size), static_cast<int>(erosion_size) ) );
	// 	cv::erode( src, dst, element );
	// 	return ret;
	// }, "erode_image");
	//
	// static operation_t op_dilate([](const image_t a, float dilate_size) {
	// 	image_t ret{};
	// 	dilate_size = std::min(std::max(dilate_size, 0.0f), 21.0f);
	// 	const cv::Mat src{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, a.as_void_const()};
	// 	cv::Mat dst{IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, CV_32F, ret.get_data().data.data()};
	// 	const cv::Mat element = cv::getStructuringElement( cv::MORPH_CROSS,
	// 						 cv::Size( static_cast<int>(2*dilate_size + 1), static_cast<int>(2*dilate_size+1) ),
	// 						 cv::Point( static_cast<int>(dilate_size), static_cast<int>(dilate_size) ) );
	// 	cv::dilate( src, dst, element );
	// 	return ret;
	// }, "erode_image");

	operator_builder builder{};
	builder.build(op_image_ephemeral, make_add<image_t>(), make_sub<image_t>(), make_mul<image_t>(), make_div<image_t>(), op_image_x, op_image_y,
				op_image_sin, op_image_gt, op_image_lt, op_image_cos, op_image_log, op_image_exp, op_image_or, op_image_and, op_image_xor,
				op_image_cos_off, op_image_sin_off, op_image_perlin, op_image_noise, op_image_random, op_image_2d_perlin_eph, op_image_not,
				op_image_grad, op_image_2d_perlin_oct);
	// builder.build(op_image_grad, op_image_x, op_image_y);
	program->set_operations(builder.grab());
}

void setup_gp_system(const blt::size_t population_size)
{
	reference_image = image_storage_t::from_file("../silly.png");

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
	images_red.resize(population_size);
	images_green.resize(population_size);
	images_blue.resize(population_size);

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
		const auto avg = stats.average_fitness.load(std::memory_order_relaxed);
		const auto best = stats.best_fitness.load(std::memory_order_relaxed);
		const auto worst = stats.worst_fitness.load(std::memory_order_relaxed);
		const auto overall = stats.overall_fitness.load(std::memory_order_relaxed);

		average_fitness.push_back(static_cast<float>(avg));
		best_fitness.push_back(static_cast<float>(best));
		worst_fitness.push_back(static_cast<float>(worst));
		overall_fitness.push_back(static_cast<float>(overall));

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

std::array<image_ipixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3>& get_image(const blt::size_t index)
{
	for (const auto& [i, image_red, image_green, image_blue] : blt::zip(images_red[index], images_green[index],
																				images_blue[index]).enumerate().flatten())
	{
		images[index][i * 3] = image_red;
		images[index][i * 3 + 1] = image_green;
		images[index][i * 3 + 2] = image_blue;
	}
	return images[index];
}

void cleanup()
{
	for (const auto program : programs)
		delete program;
}

const std::array<image_storage_t, 3>& get_reference_image()
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
			// image_data[(x * IMAGE_DIMENSIONS + y) * 3 + 0] = std::pow(image[0].get(x, y), 1.0f / 2.2f);
			// image_data[(x * IMAGE_DIMENSIONS + y) * 3 + 1] = std::pow(image[1].get(x, y), 1.0f / 2.2f);
			// image_data[(x * IMAGE_DIMENSIONS + y) * 3 + 2] = std::pow(image[2].get(x, y), 1.0f / 2.2f);
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
	if (size > images_red.size())
		images_red.resize(size);
	if (size > images_green.size())
		images_green.resize(size);
	if (size > images_blue.size())
		images_blue.resize(size);
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
{}

std::tuple<const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&> get_fitness_history()
{
	return {average_fitness, best_fitness, worst_fitness, overall_fitness};
}

std::array<population_t*, 3> get_populations()
{
	return {&programs[0]->get_current_pop(), &programs[1]->get_current_pop(), &programs[2]->get_current_pop()};
}

void set_use_gamma_correction(bool use)
{use_gamma_correction = true;}
