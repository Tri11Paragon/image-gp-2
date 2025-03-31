#include <iostream>
#include <operations.h>
#include <random>
#include <blt/gp/program.h>
#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/camera.h"
#include <imgui.h>
#include <image_storage.h>

using namespace blt::gp;

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::batch_renderer_2d renderer_2d(resources, global_matrices);
blt::gfx::first_person_camera camera;

gp_program program{
	[]() {
		return std::random_device()();
	}
};

void setup_operations()
{
	static operation_t op_sin([](const float a) {
		return std::sin(a);
	}, "sin");
	static operation_t op_cos([](const float a) {
		return std::cos(a);
	}, "cos");
	static operation_t op_exp([](const float a) {
		return std::exp(a);
	}, "exp");
	static operation_t op_log([](const float a) {
		return a <= 0.0f ? 0.0f : std::log(a);
	}, "log");
	static auto lit = operation_t([]() {
		return program.get_random().get_float(-1.0f, 1.0f);
	}, "lit").set_ephemeral();

	// static gp:: operation_t op_x([](const context& context)
	// {
	// 	return context.x;
	// }, "x");

	operator_builder builder{};
	builder.build(
		make_add<float>(),
		make_sub<float>(),
		make_mul<float>(),
		make_prot_div<float>(),
		op_sin, op_cos, op_exp, op_log, lit
		);
	program.set_operations(builder.grab());
}

void init(const blt::gfx::window_data&)
{
	using namespace blt::gfx;


	global_matrices.create_internals();
	resources.load_resources();
	renderer_2d.create();
}

void update(const blt::gfx::window_data& data)
{
	global_matrices.update_perspectives(data.width, data.height, 90, 0.1, 2000);

	camera.update();
	camera.update_view(global_matrices);
	global_matrices.update();

	renderer_2d.render(data.width, data.height);
}

void destroy(const blt::gfx::window_data&)
{
	global_matrices.cleanup();
	resources.cleanup();
	renderer_2d.cleanup();
	blt::gfx::cleanup();
}

int main()
{
	blt::gfx::init(blt::gfx::window_data{"Image GP", init, update, destroy}.setSyncInterval(1));
}