#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
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

#ifndef IMAGE_STORAGE_H
#define IMAGE_STORAGE_H

#include <array>
#include <atomic>
#include <string>
#include <blt/logging/logging.h>
#include <blt/std/types.h>
#include <mutex>
#include <blt/std/hashmap.h>

#ifndef BLT_IMAGE_SIZE
#define BLT_IMAGE_SIZE 256
#endif

using image_pixel_t = float;
constexpr blt::i32 IMAGE_DIMENSIONS = BLT_IMAGE_SIZE;
constexpr blt::i32 IMAGE_CHANNELS = 1;

constexpr blt::size_t IMAGE_SIZE = IMAGE_DIMENSIONS * IMAGE_DIMENSIONS;
constexpr blt::size_t IMAGE_SIZE_CHANNELS = IMAGE_SIZE * IMAGE_CHANNELS;
constexpr blt::size_t IMAGE_SIZE_BYTES = IMAGE_SIZE_CHANNELS * sizeof(image_pixel_t);

struct image_storage_t
{
	std::array<image_pixel_t, IMAGE_SIZE_CHANNELS> data;

	static std::array<image_storage_t, 3> from_file(const std::string& path);

	image_pixel_t& get(const blt::size_t x, const blt::size_t y)
	{
		return data[(y * IMAGE_DIMENSIONS + x) * IMAGE_CHANNELS];
	}

	[[nodiscard]] const image_pixel_t& get(const blt::size_t x, const blt::size_t y) const
	{
		return data[(y * IMAGE_DIMENSIONS + x) * IMAGE_CHANNELS];
	}

	void normalize();
};

inline std::atomic_uint64_t g_allocated_blocks = 0;
inline std::atomic_uint64_t g_deallocated_blocks = 0;

inline std::mutex g_image_list_mutex;

struct image_cleaner_t
{
	~image_cleaner_t()
	{
		for (auto v : images)
			delete v;
	}

	std::vector<image_storage_t*> images;
};

inline image_cleaner_t g_image_list;

struct image_t
{
	explicit image_t()
	{
		image_storage_t* front = nullptr;
		{
			std::scoped_lock lock(g_image_list_mutex);
			if (!g_image_list.images.empty())
			{
				front = g_image_list.images.back();
				g_image_list.images.pop_back();
			}
		}
		if (front)
			data = front;
		else
			data = new image_storage_t;
		++g_allocated_blocks;
	}

	void drop()
	{
		{
			std::scoped_lock lock(g_image_list_mutex);
			g_image_list.images.push_back(data);
		}
		data = nullptr;
		++g_deallocated_blocks;
	}

	[[nodiscard]] void* as_void_const() const
	{
		return const_cast<void*>(static_cast<const void*>(data->data.data()));
	}

	[[nodiscard]] void* as_void() const
	{
		return data->data.data();
	}

	void normalize() const
	{
		data->normalize();
	}

	friend image_t operator+(const image_t& lhs, const image_t& rhs);

	friend image_t operator-(const image_t& lhs, const image_t& rhs);

	friend image_t operator*(const image_t& lhs, const image_t& rhs);

	friend image_t operator/(const image_t& lhs, const image_t& rhs);

	image_storage_t& get_data()
	{
		return *data;
	}

	[[nodiscard]] const image_storage_t& get_data() const
	{
		return *data;
	}

private:
	image_storage_t* data;
};

#endif //IMAGE_STORAGE_H
