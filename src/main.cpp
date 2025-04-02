#include <gp_system.h>

#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/camera.h"
#include <imgui.h>

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::batch_renderer_2d renderer_2d(resources, global_matrices);
blt::gfx::first_person_camera camera;

void init(const blt::gfx::window_data&)
{
	using namespace blt::gfx;

	setup_gp_system(64);

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