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
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <blt/iterator/zip.h>
#include <blt/math/vectors.h>

image_storage_t image_storage_t::from_file(const std::string& path)
{
	stbi_set_flip_vertically_on_load(true);
	int x, y, channels;
	auto* data = stbi_loadf(path.c_str(), &x, &y, &channels, 4);

	image_storage_t storage{};
	stbir_resize_float_linear(data, x, y, 0, storage.data.data(), IMAGE_DIMENSIONS, IMAGE_DIMENSIONS, 0, STBIR_RGBA);

	STBI_FREE(data);
	return storage;
}

image_t operator/(const image_t& lhs, const image_t& rhs)
{
	const image_t ret;
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = blt::f_equal(r, 0) ? 0 : l / r;
	return ret;
}

image_t operator*(const image_t& lhs, const image_t& rhs)
{
	const image_t ret;
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = l * r;
	return ret;
}

image_t operator-(const image_t& lhs, const image_t& rhs)
{
	const image_t ret;
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = l - r;
	return ret;
}

image_t operator+(const image_t& lhs, const image_t& rhs)
{
	const image_t ret;
	for (auto [ref, l, r] : blt::zip(ret.data->data, lhs.data->data, rhs.data->data))
		ref = l + r;
	return ret;
}
