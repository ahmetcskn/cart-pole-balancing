#!/usr/bin/env python3
"""PyTorch PPO, existing C++ physics, and a live training viewer.

python3 train.py                      # watch every training episode
python3 train.py --speed 20            # accelerate the same training episodes
python3 train.py --headless --steps 200000
python3 train.py --play                # watch policy.pt
python3 train.py --play --model model_v2.pt  # watch a named policy
python3 train.py --task swingup        # learn from the original theta=0.8 reset
python3 train.py --self-test           # check C++ bridge and physics consistency

Only this project receives generated files: .rl_build/ and rl_runs/.
No downloads, dependency installation, or modification of existing sources.
Dependencies: Python 3, PyTorch, NumPy, pybind11, GLFW/OpenGL/GLUT, CMake, and a C++17 compiler.
C++ bindings live in src/python_bridge.cpp and are built with CMake.
Rendering uses the existing C++ viewer(glfw). Training metrics are printed to the terminal.
"""
from pathlib import Path
import argparse
import sys
import pybind11
import csv
import math
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parent
DT = 0.04  # four existing 0.01-second C++ steps per policy decision
LIMIT = 8.0
HORIZON = 500  # 20 simulated seconds
FORCE = 30.0  # Only horizontal cart force; pendulum torque is always zero.

def load_bridge():
    """Build and import the C++ classes exposed through pybind11."""
    build_dir = ROOT / '.rl_build/pybind11'
    subprocess.run([
        'cmake', '-S', str(ROOT), '-B', str(build_dir),
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DPython_EXECUTABLE={sys.executable}',
        f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',
    ], check=True, cwd=ROOT)
    subprocess.run([
        'cmake', '--build', str(build_dir), '--target', 'stick_physics',
    ], check=True, cwd=ROOT)
    sys.path.insert(0, str(build_dir))
    import stick_physics
    return stick_physics


class Environment:
    def __init__(self, lib, task, seed):
        self.lib, self.task = lib, task
        self.simulation = lib.CartPole()
        self.rng = np.random.default_rng(seed)
        self.substeps = round(DT / self.simulation.parameters().dt)
        if self.substeps < 1 or not math.isclose(self.substeps * self.simulation.parameters().dt, DT):
            raise ValueError('C++ dt must be compatible with the 0.04-second control interval.')
        self.length = self.simulation.parameters().length
        self.count = 0

    def close(self):
        self.simulation = None  # pybind11 releases the C++ object automatically.

    def state(self):
        s = self.simulation.state()
        return np.array([s.x, s.velocity, s.theta, s.omega], dtype=np.float64)

    def observe(self):
        x, v, theta, omega = self.state()
        return np.array([x / LIMIT, v / 5, math.sin(theta), math.cos(theta), omega / 8], dtype=np.float32)

    def reset(self):
        # Balance starts close to the unstable upright equilibrium.
        # Swingup keeps the original reset angle and adds a small random perturbation.
        theta = (math.pi if self.task == 'balance' else 0.8) + self.rng.uniform(-0.18, 0.18)
        self.simulation.reset(self.rng.uniform(-0.15, 0.15),
                          self.rng.uniform(-0.1, 0.1), theta, self.rng.uniform(-0.1, 0.1))
        self.count = 0
        return self.observe()

    def step(self, action):
        action = np.clip(action, -1, 1)
        for _ in range(self.substeps):
            self.simulation.step(float(action[0] * FORCE))
        self.count += 1
        s = self.state()
        if not np.isfinite(s).all():
            raise FloatingPointError('The C++ simulation produced a non-finite state.')
        x, v, theta, omega = s
        error = math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))
        effort = float(np.square(action).mean())
        if self.task == 'balance':
            reward = 1 - (error / 0.7) ** 2 - 0.1 * (x / LIMIT) ** 2 - 0.01 * effort - 0.001 * v ** 2
        else:
            reward = (1 - math.cos(theta)) / 2 - 0.1 * (x / LIMIT) ** 2 - 0.002 * omega ** 2 - 0.01 * effort
        reason = 'track limit' if abs(x) > LIMIT else ''
        if self.task == 'balance' and abs(error) > 0.7:
            reason = 'pole fell'
        terminated = bool(reason)
        if terminated:
            reward -= 5.0
        truncated = self.count >= HORIZON and not terminated
        return self.observe(), reward, terminated, truncated, reason or ('time limit reached' if truncated else '')


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.full((1,), -0.5))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-3, 1).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class Viewer:
    """Calls the existing C++ GLFW viewer; there is no Python drawing code."""
    def __init__(self, lib, speed):
        self.window = lib.Viewer()
        self.closed = False
        self.speed = speed
        self.last = time.monotonic()

    def close(self):
        if not self.closed:
            self.window.close()
            self.closed = True

    def pump(self):
        if self.closed or not self.window.poll():
            self.close()
            raise KeyboardInterrupt

    def render(self, env):
        self.pump()
        self.window.render(env.simulation)
        deadline = self.last + DT / self.speed
        while time.monotonic() < deadline:
            self.pump()
            time.sleep(0.001)
        self.last = time.monotonic()


def update(model, optimizer, batch):
    obs, latent, old_logp, returns, advantages = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in batch]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    for _ in range(10):
        for ids in torch.randperm(len(obs)).split(128):
            dist = model.distribution(obs[ids])
            logp = dist.log_prob(latent[ids]).sum(-1)
            # Latent Gaussian log ratios are also the tanh-action log ratios:
            # the fixed transform's Jacobian cancels between old and new policies.
            ratio = (logp - old_logp[ids]).exp()
            adv = advantages[ids]
            actor_loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            value_loss = (model.value(obs[ids]) - returns[ids]).square().mean()
            loss = actor_loss + 0.5 * value_loss - 0.001 * dist.entropy().sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def evaluate(lib, model, task, seed, episodes=10):
    env = Environment(lib, task, seed)
    totals, durations = [], []
    try:
        for _ in range(episodes):
            obs, total = env.reset(), 0
            for _ in range(HORIZON):
                with torch.no_grad():
                    action = model.actor(torch.from_numpy(obs)).tanh().numpy()
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                if term or trunc:
                    break
            totals.append(total)
            durations.append(env.count * DT)
    finally:
        env.close()
    return float(np.mean(totals)), float(np.mean(durations))


def self_test(lib):
    env = Environment(lib, 'balance', 1)
    try:
        policy = Policy()
        assert policy.actor(torch.zeros(5)).shape == (1,)
        assert policy.distribution(torch.zeros(5)).sample().shape == (1,)
        env.simulation.reset(0, 0, 0.8, 0)
        theta, force = 0.8, 2.0
        a11, a12, a21, a22 = 2.5, math.cos(theta), 0.25 * math.cos(theta), 1.1
        b1, b2 = -9.81 * math.sin(theta), force
        det = a11 * a22 - a12 * a21
        alpha, accel = (b1 * a22 - b2 * a12) / det, (a11 * b2 - b1 * a21) / det
        env.simulation.step(force)
        np.testing.assert_allclose(env.state(), [accel * .0001, accel * .01, theta + alpha * .0001, alpha * .01], atol=1e-12)
        env.simulation.reset(0, 0, math.pi, 0)
        _, r, term, trunc, _ = env.step(np.zeros(1))
        assert r > .99 and not term and not trunc
        env.simulation.reset(LIMIT + 0.1, 0, math.pi, 0)
        assert env.step(np.zeros(1))[2]
        env.simulation.reset(0, 0, math.pi, 0)
        env.count = HORIZON - 1
        _, _, term, trunc, _ = env.step(np.zeros(1))
        assert trunc and not term
        env.simulation.reset(0, 0, math.pi + .8, 0)
        assert env.step(np.zeros(1))[2]
        print('OK: C++ step, state transfer, upright reward, fall/track penalty, and time limit.')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=['balance', 'swingup'], default='balance')
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--speed', type=float, default=4.0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--model', help='Model filename to load with --play; example: model_v2.pt')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.model is not None:
        if not args.play:
            parser.error('--model can only be used with --play.')
        if Path(args.model).name != args.model or not args.model.endswith('.pt'):
            parser.error('--model requires a .pt filename only; example: model_v2.pt')
    if args.steps < 1 or not .5 <= args.speed <= 30:
        parser.error('--steps must be positive and --speed must be between 0.5 and 30.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    lib = load_bridge()
    if args.self_test:
        self_test(lib)
        return
    model = Policy()
    run_dir = ROOT / 'rl_runs' / args.task
    checkpoint = run_dir / (args.model or 'policy.pt')
    if args.play:
        if not checkpoint.exists():
            parser.error(f'Model file not found: {checkpoint}')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(payload['model'])
    if args.play and args.headless:
        score, duration = evaluate(lib, model, args.task, args.seed + 10000)
        print(f'10 independent episodes: mean reward={score:.2f}, mean duration={duration:.2f}s')
        return
    viewer = None
    if not args.headless:
        try:
            viewer = Viewer(lib, args.speed)
        except Exception as exc:
            parser.exit(1, f'Could not open the viewer window: {exc}\nHeadless training: python3 train.py --headless\n')
    env = Environment(lib, args.task, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    steps, episode, total = 0, 1, 0.0
    obs = env.reset()
    output = None
    try:
        if not args.play:
            run_dir.mkdir(parents=True, exist_ok=True)
            output = (run_dir / f'training-{time.time_ns()}.csv').open('w', newline='')
            log = csv.writer(output)
            log.writerow(['episode', 'step', 'return', 'seconds', 'end_reason'])
            score, duration = evaluate(lib, model, args.task, args.seed + 10000)
            print(f'Before training: reward={score:.2f}, duration={duration:.2f}s', flush=True)
        while args.play or steps < args.steps:
            observations, actions, logps, values, rewards, next_values, boundaries = [], [], [], [], [], [], []
            for _ in range(1024 if args.play else min(1024, args.steps - steps)):
                with torch.no_grad():
                    tensor = torch.from_numpy(obs)
                    dist = model.distribution(tensor)
                    latent = dist.mean if args.play else dist.sample()
                    action = latent.tanh().numpy()
                    value, logp = model.value(tensor).item(), dist.log_prob(latent).sum().item()
                next_obs, reward, term, trunc, reason = env.step(action)
                with torch.no_grad():
                    next_value = 0.0 if term else model.value(torch.from_numpy(next_obs)).item()
                observations.append(obs)
                actions.append(latent.numpy())
                logps.append(logp)
                values.append(value)
                rewards.append(reward)
                next_values.append(next_value)
                boundaries.append(term or trunc)
                steps += 1
                total += reward
                if viewer:
                    viewer.render(env)
                obs = next_obs
                if term or trunc:
                    print(f'Episode {episode:5d} | step {steps:7d} | reward {total:8.2f} | {env.count * DT:5.2f}s | {reason}', flush=True)
                    if output:
                        log.writerow([episode, steps, total, env.count * DT, reason])
                        output.flush()
                    if viewer:
                        deadline = time.monotonic() + .3 / max(1, viewer.speed / 4)
                        while time.monotonic() < deadline:
                            viewer.pump()
                            time.sleep(.01)
                    obs, total, episode = env.reset(), 0.0, episode + 1
            if not args.play:
                advantages, gae = np.zeros(len(rewards), dtype=np.float32), 0.0
                for i in reversed(range(len(rewards))):
                    delta = rewards[i] + .99 * next_values[i] - values[i]
                    # Time limits bootstrap value, but GAE never crosses a reset.
                    gae = delta + .99 * .95 * (not boundaries[i]) * gae
                    advantages[i] = gae
                update(model, optimizer, [observations, actions, logps, advantages + np.asarray(values), advantages])
                save(model, optimizer, checkpoint, args, steps)
                if viewer:
                    viewer.pump()
    except KeyboardInterrupt:
        print('\nStopped.', flush=True)
    finally:
        if not args.play:
            save(model, optimizer, checkpoint, args, steps)
            print(f'Model saved: {checkpoint}', flush=True)
        if output:
            output.close()
        env.close()
        if viewer and not viewer.closed:
            viewer.close()
    if not args.play:
        score, duration = evaluate(lib, model, args.task, args.seed + 10000)
        print(f'After training (10 independent episodes): reward={score:.2f}, duration={duration:.2f}s', flush=True)


def save(model, optimizer, path, args, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'task': args.task, 'seed': args.seed, 'steps': steps,
                'force_limit': FORCE, 'action_space': 'horizontal_force_only', 'control_dt': DT}, temporary)
    temporary.replace(path)


if __name__ == '__main__':
    main()
