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

#ifndef GP_SYSTEM_H
#define GP_SYSTEM_H
#include <image_storage.h>
#include <blt/std/types.h>

void setup_gp_system(blt::size_t population_size);

void run_step();

bool should_terminate();

blt::u32 get_generation();

void reset_programs();

void regenerate_image(blt::size_t index, float& image_storage, blt::i32 width, blt::i32 height);

void set_population_size(blt::u32 size);

std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3>& get_image(blt::size_t index);

void cleanup();

std::array<image_storage_t, 3>& get_reference_image();

std::array<image_pixel_t, IMAGE_DIMENSIONS * IMAGE_DIMENSIONS * 3> to_gl_image(const std::array<image_storage_t, 3>& image);

std::tuple<const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&> get_fitness_history();

#endif //GP_SYSTEM_H
