# Product Finder — OpenEnv RL Environment

> An AI environment where an agent learns to choose the best product based on price, rating, and delivery speed — built using the [OpenEnv](https://github.com/openenv) framework for reinforcement learning environments.

<p align="center">
  <img src="https://img.shields.io/badge/status-active-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/python-95%25-blue?style=flat-square&logo=python&logoColor=white"/>
</p>

## ✨ What It Does

This is a custom reinforcement learning environment that simulates a product-selection decision problem. An agent is presented with a set of products, each with a **price**, **rating**, and **delivery speed**, and must learn to pick the best option according to a reward signal — similar in structure to how an e-commerce recommendation or comparison-shopping agent might be trained.

The environment ships with **three difficulty tiers** — easy, medium, and hard — letting an agent (or a human testing the env) progress from simple, clearly-dominant choices to harder tradeoffs where no single product wins on every dimension. Rewards are normalized between **0 and 1**, scaled by how close the agent's choice is to the optimal product given the task's weighting of price vs. rating vs. delivery speed.

## 🧠 Why I Built It

I wanted hands-on experience with the OpenEnv environment spec rather than just using pre-built Gym environments — building one from scratch forces you to actually think through state representation, reward shaping, and task difficulty scaling instead of treating them as already solved.

## 🛠️ Tech Stack

- **Language:** Python
- **Environment definition:** `product_env.py` — core environment logic, state/action space, reward function
- **Serving:** `server.py` — exposes the environment for agent interaction
- **Inference:** `inference.py` — runs a trained/test agent against the environment
- **Config:** `openenv.yaml` — environment spec/configuration
- **Containerization:** `Dockerfile` — for reproducible environment deployment

## 🏗️ How It Works

1. `product_env.py` defines the environment: the product attributes (price, rating, delivery speed), the action space (which product to select), and the reward function (0–1 scale based on selection quality)
2. Three task difficulty levels adjust how clearly one product dominates the others, testing whether an agent can handle ambiguous tradeoffs, not just easy wins
3. `server.py` exposes this environment so an agent (trained separately or via `inference.py`) can interact with it in the standard OpenEnv request/response pattern
4. The Dockerfile packages the whole thing for consistent execution outside the dev environment

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Docker (optional, for containerized run)

### Installation
```bash
git clone https://github.com/rithwikkr0/product-finder-openenv.git
cd product-finder-openenv
pip install -r requirements.txt   # add this file if not already present
```

### Running It
```bash
python server.py
# in a separate process/terminal:
python inference.py
```

Or via Docker:
```bash
docker build -t product-finder-openenv .
docker run product-finder-openenv
```

## 🗺️ Roadmap / Known Limitations

- [ ] Add a `requirements.txt` if not already present, so installation actually works for someone cloning this fresh
- [ ] Document the exact reward formula (current README assumption: weighted combination of price/rating/delivery-speed deltas — confirm and specify)
- [ ] Add example output / a sample agent run showing reward progression across difficulty tiers
- [ ] Consider adding a baseline agent (random or greedy) for comparison against any trained policy

## 📄 License

MIT (or your choice) — add a LICENSE file at repo root if missing.

---
*Built by [Rithwik K R](https://github.com/rithwikkr0)*
