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

using image_pixel_t = float;
constexpr blt::i32 IMAGE_DIMENSIONS = 256;
constexpr blt::i32 IMAGE_CHANNELS = 4;

constexpr blt::size_t IMAGE_SIZE = IMAGE_DIMENSIONS * IMAGE_DIMENSIONS;
constexpr blt::size_t IMAGE_SIZE_CHANNELS = IMAGE_SIZE * IMAGE_CHANNELS;
constexpr blt::size_t IMAGE_SIZE_BYTES = IMAGE_SIZE_CHANNELS * sizeof(image_pixel_t);

struct image_storage_t
{
	std::array<image_pixel_t, IMAGE_SIZE_CHANNELS> data;

	static image_storage_t from_file(const std::string& path);

	image_pixel_t& get(const blt::size_t x, const blt::size_t y, const blt::i32 c)
	{
		return data[(y * IMAGE_DIMENSIONS + x) * IMAGE_CHANNELS + c];
	}

	[[nodiscard]] const image_pixel_t& get(const blt::size_t x, const blt::size_t y, const blt::i32 c) const
	{
		return data[(y * IMAGE_DIMENSIONS + x) * IMAGE_CHANNELS + c];
	}
};

inline std::atomic_uint64_t g_allocated_nodes = 0;
inline std::atomic_uint64_t g_deallocated_nodes = 0;

struct atomic_node_t
{
	explicit atomic_node_t(image_storage_t* data): data(data)
	{
		++g_allocated_nodes;
	}

	std::atomic<atomic_node_t*> next = nullptr;
	image_storage_t* data = nullptr;

	~atomic_node_t()
	{
		++g_deallocated_nodes;
	}
};

inline std::atomic_uint64_t g_allocated_blocks = 0;
inline std::atomic_uint64_t g_deallocated_blocks = 0;

class atomic_list_t
{
public:
	atomic_list_t() = default;

	void push_back(atomic_node_t* node)
	{
		while (true)
		{
			auto head = this->m_head.load();
			node->next = head;
			if (this->m_head.compare_exchange_weak(head, node))
				break;
		}
	}

	atomic_node_t* pop_front()
	{
		while (true)
		{
			auto head = this->m_head.load();
			if (head == nullptr)
				return nullptr;
			if (this->m_head.compare_exchange_weak(head, head->next))
				return head;
		}
	}

	~atomic_list_t()
	{
		while (m_head != nullptr)
			delete pop_front();
	}

private:
	std::atomic<atomic_node_t*> m_head = nullptr;
};

inline atomic_list_t g_image_list;

struct image_t
{
	image_t()
	{
		const auto front = g_image_list.pop_front();
		if (front)
		{
			data = front->data;
			delete front;
		} else
			data = new image_storage_t;
		++g_allocated_blocks;
	}

	void drop()
	{
		const auto node = new atomic_node_t(data); // NOLINT
		data = nullptr;
		g_image_list.push_back(node);
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
