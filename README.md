# Cart-Pole Balancing

A C++ cart-pole simulation with PyTorch reinforcement learning. The model learns to balance the pole by moving the cart along the x-axis. The RL controller applies no torque.

Physics and GLFW rendering run in C++. Python handles PPO training and connects to C++ through pybind11. You can watch training live in the GLFW window.

## Requirements

C++17, CMake, Python 3, PyTorch, NumPy, pybind11, and GLFW/OpenGL/GLUT development libraries, including Python development headers.

## Usage

Run these commands from the project directory. The C++ module is built automatically.

```bash
# Start training from scratch
python3 train.py

# Watch the saved model
python3 train.py --play

# Watch a named model (use the filename only)
python3 train.py --play --model model_v2.pt

# Train without opening a window
python3 train.py --headless

# Check physics and bindings without training
python3 train.py --self-test
```

Press `Ctrl+C` or close the window to stop.

## Saved models

By default, training saves to `rl_runs/balance/policy.pt`. Each new training run starts from scratch and overwrites this file. To keep an older model, copy it under a different name first.

`--play` runs the model without training or updating its weights.

## Files

- `src/cart_pole.cpp`: physics engine
- `src/viewer.cpp`: GLFW rendering
- `src/python_bridge.cpp`: pybind11 bindings
- `train.py`: neural networks and PPO training
