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

struct image_storage_t
{
	std::array<float, 4> data;
};

struct atomic_node_t
{
	std::atomic<atomic_node_t*> next = nullptr;
	image_storage_t* data = nullptr;
};

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
		}else
			data = new image_storage_t;
	}

	void drop()
	{
		const auto node = new atomic_node_t(); // NOLINT
		node->data = data;
		data = nullptr;
		g_image_list.push_back(node);
	}
private:
	image_storage_t* data;
};

#endif //IMAGE_STORAGE_H
