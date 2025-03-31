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

#include <blt/gp/program.h>

#ifndef OPERATIONS_H
#define OPERATIONS_H

template <typename T>
auto& make_add()
{
	static blt::gp::operation_t add{
		[](const T a, const T b) -> T {
			return a + b;
		},
		"add"
	};
	return add;
}

template <typename T>
auto& make_sub()
{
	static blt::gp::operation_t sub([](const T a, const T b) -> T {
		return a - b;
	}, "sub");
	return sub;
}

template <typename T>
auto& make_mul()
{
	static blt::gp::operation_t mul([](const T a, const T b) -> T {
		return a * b;
	}, "mul");
	return mul;
}

template <typename T>
auto& make_prot_div()
{
	static blt::gp::operation_t pro_div([](const T a, const T b) -> T {
		return b == static_cast<T>(0) ? static_cast<T>(0) : a / b;
	}, "div");
	return pro_div;
}

#endif //OPERATIONS_H
