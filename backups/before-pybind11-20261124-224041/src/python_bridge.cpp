// C interface for Python ctypes; physics stays in CartPole.
#include "cart_pole.hpp"
extern "C" {
void* cp_create() { return new CartPole(); }
void cp_destroy(void* p) { delete static_cast<CartPole*>(p); }
void cp_reset(void* p, double x, double v, double theta, double omega) {
    auto& sim = *static_cast<CartPole*>(p);
    sim.reset();
    // The underlying object is mutable; only its public reference is const.
    // This adapter adds reset-state control without editing the user's header.
    auto& s = const_cast<CartPoleState&>(sim.state());
    s.x = x; s.velocity = v; s.theta = theta; s.omega = omega;
}
void cp_step(void* p, double force) {
    static_cast<CartPole*>(p)->step(force, 0.0);
}
void cp_state(void* p, double* out) {
    const auto& s = static_cast<CartPole*>(p)->state();
    out[0] = s.x; out[1] = s.velocity; out[2] = s.theta; out[3] = s.omega;
}
double cp_dt(void* p) { return static_cast<CartPole*>(p)->parameters().dt; }
double cp_length(void* p) { return static_cast<CartPole*>(p)->parameters().length; }
}
