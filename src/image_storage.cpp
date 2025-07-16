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
#include <image_storage.h>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <blt/iterator/zip.h>
#include <blt/math/vectors.h>

inline float srgb_to_linear(const float v) noexcept
{
	return (v <= 0.04045f) ? (v / 12.92f)
						   : std::pow((v + 0.055f) / 1.055f, 2.4f);
}


std::array<image_storage_t, 3> image_storage_t::from_file(const std::string& path)
{
	stbi_set_flip_vertically_on_load(true);
	int x, y, channels;
	auto* data = stbi_load(path.c_str(), &x, &y, &channels, 4);

	unsigned char* resized = nullptr;

	if (x == IMAGE_DIMENSIONS && y == IMAGE_DIMENSIONS)
	{
		resized = data;
		data = nullptr;
	} else
		resized = stbir_resize_uint8_srgb(data, x, y, 0, nullptr, IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, 0, STBIR_RGBA);

	image_storage_t storage_r{};
	image_storage_t storage_g{};
	image_storage_t storage_b{};

	for (blt::size_t i = 0; i < IMAGE_DIMENSIONS; ++i)
	{
		for (blt::size_t j = 0; j < IMAGE_DIMENSIONS; ++j)
		{
			storage_r.get(i, j) = static_cast<float>(resized[(i * IMAGE_DIMENSIONS + j) * 4]) / 255.0f;
			storage_g.get(i, j) = static_cast<float>(resized[(i * IMAGE_DIMENSIONS + j) * 4 + 1]) / 255.0f;
			storage_b.get(i, j) = static_cast<float>(resized[(i * IMAGE_DIMENSIONS + j) * 4 + 2]) / 255.0f;
		}
	}

	stbi_image_free(data);
	stbi_image_free(resized);
	return {storage_r, storage_g, storage_b};
}

std::array<image_storage_t, 3> image_istorage_t::from_file(const std::string& path)
{
	BLT_ERROR("NOT IMPLEMENTED!");
}

void image_istorage_t::normalize()
{}

void image_storage_t::normalize()
{
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();

	for (auto& pixel : data)
	{
		if (std::isnan(pixel) || std::isinf(pixel))
			pixel = 0;
		if (pixel > max)
			max = pixel;
		if (pixel < min)
			min = pixel;
	}

	for (auto& pixel : data)
		pixel = (pixel - min) / (max - min);
}

image_t operator/(const image_t& lhs, const image_t& rhs)
{
	const image_t ret{};
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = blt::f_equal(r, 0) ? 0 : l / r;
	return ret;
}

image_t operator*(const image_t& lhs, const image_t& rhs)
{
	const image_t ret{};
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = l * r;
	return ret;
}

image_t operator-(const image_t& lhs, const image_t& rhs)
{
	const image_t ret{};
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = l - r;
	return ret;
}

image_t operator+(const image_t& lhs, const image_t& rhs)
{
	const image_t ret{};
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = l + r;
	return ret;
}
