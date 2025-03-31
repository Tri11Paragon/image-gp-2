#include <iostream>
#include <operations.h>
#include <random>
#include <blt/gp/program.h>

using namespace blt;

gp::gp_program program{
	[]() {
		return std::random_device()();
	}
};

void setup_operations()
{
	static gp::operation_t op_sin([](const float a) {
		return std::sin(a);
	}, "sin");
	static gp::operation_t op_cos([](const float a) {
		return std::cos(a);
	}, "cos");
	static gp::operation_t op_exp([](const float a) {
		return std::exp(a);
	}, "exp");
	static gp::operation_t op_log([](const float a) {
		return a <= 0.0f ? 0.0f : std::log(a);
	}, "log");
	static auto lit = gp::operation_t([this]() {
		return program.get_random().get_float(-1.0f, 1.0f);
	}, "lit").set_ephemeral();

	// static gp:: operation_t op_x([](const context& context)
	// {
	// 	return context.x;
	// }, "x");

	gp::operator_builder builder{};
	builder.build(make_add<float>(), make_sub<float>(), make_mul<float>(), make_prot_div<float>(), op_sin, op_cos, op_exp, op_log, lit);
	program.set_operations(builder.grab());
}

int main()
{}
