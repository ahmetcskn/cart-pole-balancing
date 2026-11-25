// Python bindings: physics and drawing remain in their existing C++ sources.
#include "cart_pole.hpp"
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

// Existing viewer functions stay private to C++; Python never handles pointers.
extern "C" void* viewer_open();
extern "C" int viewer_poll(void* window);
extern "C" void viewer_render(void* window, void* simulation);
extern "C" void viewer_close(void* window);

class TrainingViewer {
public:
    TrainingViewer() : window_(viewer_open()) {
        if (!window_) throw std::runtime_error("GLFW penceresi açılamadı.");
    }
    ~TrainingViewer() { close(); }
    TrainingViewer(const TrainingViewer&) = delete;
    TrainingViewer& operator=(const TrainingViewer&) = delete;
    bool poll() { return window_ && viewer_poll(window_); }
    void render(CartPole& simulation) {
        if (!window_) throw std::runtime_error("GLFW penceresi kapalı.");
        viewer_render(window_, &simulation);
    }
    void close() {
        if (window_) { viewer_close(window_); window_ = nullptr; }
    }
private:
    void* window_;
};

PYBIND11_MODULE(stick_physics, m) {
    py::class_<CartPoleState>(m, "CartPoleState")
        .def_readonly("x", &CartPoleState::x)
        .def_readonly("velocity", &CartPoleState::velocity)
        .def_readonly("theta", &CartPoleState::theta)
        .def_readonly("omega", &CartPoleState::omega);

    py::class_<CartPoleParameters>(m, "CartPoleParameters")
        .def_readonly("length", &CartPoleParameters::length)
        .def_readonly("gravity", &CartPoleParameters::gravity)
        .def_readonly("pendulum_mass", &CartPoleParameters::pendulum_mass)
        .def_readonly("cart_mass", &CartPoleParameters::cart_mass)
        .def_readonly("dt", &CartPoleParameters::dt);

    py::class_<CartPole>(m, "CartPole")
        .def(py::init<>())
        .def("reset", [](CartPole& sim, double x, double velocity, double theta, double omega) {
            sim.reset(CartPoleState{x, velocity, theta, omega});
        }, py::arg("x") = 0.0, py::arg("velocity") = 0.0,
           py::arg("theta") = 0.8, py::arg("omega") = 0.0)
        // The RL policy controls horizontal force only, never torque.
        .def("step", [](CartPole& sim, double force) { sim.step(force, 0.0); }, py::arg("force"))
        .def("state", &CartPole::state, py::return_value_policy::copy)
        .def("parameters", &CartPole::parameters, py::return_value_policy::copy);

    py::class_<TrainingViewer>(m, "Viewer")
        .def(py::init<>())
        .def("poll", &TrainingViewer::poll)
        .def("render", &TrainingViewer::render)
        .def("close", &TrainingViewer::close);
}
