# RedGrid Deployment Guide: Hugging Face Spaces

This guide walks through deploying the RedGrid traffic signal optimization environment on Hugging Face Spaces and connecting it with an RL agent for training.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Create a Hugging Face Space](#create-a-hugging-face-space)
3. [Repository Setup](#repository-setup)
4. [Configure Environment](#configure-environment)
5. [Deploy Server](#deploy-server)
6. [Connect RL Agent](#connect-rl-agent)
7. [Monitor & Debug](#monitor--debug)
8. [Cost Optimization](#cost-optimization)

---

## Prerequisites

### Required Accounts & Tools
- **Hugging Face Account**: Create at https://huggingface.co/join
- **HF CLI**: Install with `pip install huggingface-hub`
- **Git**: For version control
- **Local Development**: Python 3.10+, `uv` package manager
- **Optional**: Weights & Biases account for training metrics (https://wandb.ai)

### Local Testing
Before deploying to HF Spaces, verify everything works locally:

```bash
# Install dependencies
uv sync

# Run tests
python -m pytest redgrid/tests/ -v

# Start server locally
cd redgrid
python -m uvicorn server.app:app --port 8000 --host 0.0.0.0
```

Server should be accessible at `http://localhost:8000/docs` (Swagger UI).

---

## Create a Hugging Face Space

### Step 1: Create Space on HF UI

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Fill in:
   - **Owner**: Your username or organization
   - **Space name**: `redgrid-environment` (or your preferred name)
   - **License**: `MIT` or `Apache-2.0`
   - **Space SDK**: `Docker` (we'll use Docker for custom server)
   - **Space hardware**: Start with **CPU Basic** ($0/month)
     - Upgrade to **GPU** if training needs acceleration
     - Upgrade to **Duplicate Space** if you want persistent storage

4. Click **Create Space**

### Step 2: Get Space URL

Your space will be accessible at:
```
https://huggingface.co/spaces/{username}/{space_name}
```

The API endpoint for your server will be:
```
https://{username}-{space_name}.hf.space
```

---

## Repository Setup

### Step 1: Clone Repo & Add HF Remote

```bash
# Clone your repo (ensure you have push access)
git clone https://github.com/{username}/{repo}.git
cd rl-openenv-ai

# Add HF Spaces remote
huggingface-cli repo create \
  --repo-id {username}/redgrid-environment \
  --type space \
  --space-sdk docker
```

Alternatively, git add the remote directly:
```bash
git remote add hf https://huggingface.co/spaces/{username}/redgrid-environment.git
git pull hf main  # Pull initial Space files
```

### Step 2: Copy Relevant Files to Space Root

The HF Space Docker setup expects a specific structure. Copy files:

```bash
# Create Space directory structure
mkdir -p hf_space
cp -r redgrid hf_space/
cp DEPLOYMENT.md hf_space/

# Files needed in Space root
touch hf_space/Dockerfile
touch hf_space/README.md
```

---

## Configure Environment

### Create `Dockerfile`

Place this in `hf_space/Dockerfile` (or repository root):

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY redgrid/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY redgrid /app/redgrid
COPY README.md /app/

# Expose port (HF Spaces will route traffic to :7860 → your port)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/docs || exit 1

# Start server on port 7860 (HF Spaces default)
CMD ["python", "-m", "uvicorn", "redgrid.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]
```

### Create `requirements.txt` in Root

If you want HF Spaces to auto-detect dependencies:

```txt
openenv[core]>=0.2.0
fastapi>=0.115.0
uvicorn>=0.24.0
pydantic>=2.0
```

### Create Space `README.md`

```markdown
---
title: RedGrid Traffic Optimization
emoji: 🚦
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# RedGrid OpenEnv Environment

Traffic signal optimization environment running on OpenEnv framework.

## API Documentation

- **Swagger UI**: https://huggingface.co/spaces/{username}/redgrid-environment
- **OpenAPI Spec**: `/docs`

## Quick Start

### Server Status
```bash
curl https://{username}-{space_name}.hf.space/docs
```

### Reset Environment
```bash
curl -X POST https://{username}-{space_name}.hf.space/env/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

### Step Environment
```bash
curl -X POST https://{username}-{space_name}.hf.space/env/step \
  -H "Content-Type: application/json" \
  -d '{"phases": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}'
```

## Training an RL Agent

See [DEPLOYMENT.md](./DEPLOYMENT.md) for training instructions.
```

---

## Deploy Server

### Step 1: Push to HF Spaces

```bash
# Add HF Space as remote
git remote add hf https://huggingface.co/spaces/{username}/redgrid-environment.git

# Commit and push
git add Dockerfile requirements.txt README.md redgrid/
git commit -m "Deploy RedGrid environment to HF Spaces"
git push hf main
```

### Step 2: Monitor Deployment

1. Go to your Space URL: https://huggingface.co/spaces/{username}/redgrid-environment
2. Click **Logs** tab to watch the deployment
3. Wait for status to show **Running** (usually 2-5 minutes)
4. Test the API:

```bash
# Replace {username} and {space_name} with your details
SPACE_URL="https://{username}-{space_name}.hf.space"

# Health check
curl $SPACE_URL/docs

# Reset environment
curl -X POST $SPACE_URL/env/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

### Step 3: Verify API Response

Expected response from reset:
```json
{
  "current_time": 0,
  "task": "easy",
  "intersections": [...],
  "total_vehicles_active": 0,
  "total_vehicles_departed": 0,
  "total_vehicles_waiting": 0,
  "total_cumulative_wait": 0.0,
  "done": false,
  "reward": 0.0
}
```

---

## Connect RL Agent

### Option 1: Local Agent Connecting to Remote Space

Create `train_agent.py` for local training against the remote environment:

```python
"""Train RL agent against RedGrid environment on HF Spaces."""

import numpy as np
import requests
import json
from typing import Dict, Tuple, Any

class RedGridClient:
    """Client for connecting to RedGrid server on HF Spaces."""
    
    def __init__(self, base_url: str):
        """
        Args:
            base_url: e.g., "https://username-spacename.hf.space"
        """
        self.base_url = base_url.rstrip('/')
    
    def reset(self, task: str = "easy", seed: int = 42) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        response = requests.post(
            f"{self.base_url}/env/reset",
            json={"task": task, "seed": seed}
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, phases: list) -> Dict[str, Any]:
        """Execute one step with phase actions."""
        response = requests.post(
            f"{self.base_url}/env/step",
            json={"phases": phases}
        )
        response.raise_for_status()
        return response.json()
    
    def get_state_size(self, observation: Dict) -> int:
        """Extract observation size from intersection data."""
        num_intersections = len(observation.get("intersections", []))
        # Per-intersection: current_phase, phase_timer, in_yellow, 
        #                   4 queues, 4 occupancies = 11 features
        obs_size = num_intersections * 11
        return obs_size
    
    def get_action_size(self, observation: Dict) -> int:
        """Get number of intersections (action size)."""
        return len(observation.get("intersections", []))


class SimpleAgent:
    """Baseline: Random action agent for testing."""
    
    def __init__(self, action_size: int):
        self.action_size = action_size
    
    def get_action(self, observation: Dict) -> list:
        """Random phase selection [0 or 1 per intersection]."""
        return [np.random.randint(0, 2) for _ in range(self.action_size)]


def train_agent(space_url: str, task: str = "easy", num_episodes: int = 5):
    """
    Simple training loop: run episodes and collect metrics.
    
    In production, use stable-baselines3 or similar framework.
    """
    client = RedGridClient(space_url)
    agent = None
    
    episode_returns = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs = client.reset(task=task, seed=episode)
        
        # Initialize agent with observation size
        if agent is None:
            action_size = client.get_action_size(obs)
            agent = SimpleAgent(action_size)
        
        episode_reward = 0.0
        step_count = 0
        
        # Episode loop
        while True:
            # Get action (random for baseline)
            action = agent.get_action(obs)
            
            # Step environment
            obs = client.step(action)
            episode_reward += obs.get("reward", 0.0)
            step_count += 1
            
            # Check termination
            if obs.get("done", False):
                metadata = obs.get("metadata", {})
                grader_score = metadata.get("grader_score", 0.0)
                print(f"Episode {episode+1}: "
                      f"Steps={step_count}, "
                      f"Return={episode_reward:.2f}, "
                      f"Grader Score={grader_score:.3f}")
                episode_returns.append(episode_reward)
                break
    
    print(f"\nAverage Return: {np.mean(episode_returns):.2f}")
    return episode_returns


if __name__ == "__main__":
    import sys
    
    # Use command-line argument or default
    space_url = sys.argv[1] if len(sys.argv) > 1 else \
                "https://username-redgrid-environment.hf.space"
    
    print(f"Connecting to {space_url}...")
    train_agent(space_url, task="easy", num_episodes=5)
```

Run training:
```bash
python train_agent.py https://{username}-{space_name}.hf.space
```

### Option 2: Use Stable-Baselines3 (PPO/DQN)

Install:
```bash
pip install stable-baselines3 gym
```

Create `sb3_train.py`:

```python
"""Train using stable-baselines3 PPO algorithm."""

import gym
from gym import spaces
import numpy as np
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class RedGridEnv(gym.Env):
    """Gym-compatible wrapper for RedGrid on HF Spaces."""
    
    def __init__(self, space_url: str, task: str = "easy"):
        super().__init__()
        self.space_url = space_url.rstrip('/')
        self.task = task
        self.action_size = None
        self.obs_size = None
        
        # Initialize to get sizes
        obs = self._reset()
        self.action_size = len(obs["intersections"])
        self.obs_size = self.action_size * 11
        
        # Define spaces
        self.action_space = spaces.MultiDiscrete([2] * self.action_size)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.obs_size,), dtype=np.float32
        )
    
    def _reset(self):
        """Reset environment."""
        response = requests.post(
            f"{self.space_url}/env/reset",
            json={"task": self.task}
        )
        response.raise_for_status()
        return response.json()
    
    def _obs_to_vector(self, obs: dict) -> np.ndarray:
        """Convert observation dict to flat vector."""
        vector = []
        for inter_obs in obs["intersections"]:
            vector.extend([
                inter_obs["current_phase"],
                inter_obs["phase_timer"],
                int(inter_obs["in_yellow"]),
                inter_obs["queue_north"],
                inter_obs["queue_south"],
                inter_obs["queue_east"],
                inter_obs["queue_west"],
                inter_obs["occupancy_north"],
                inter_obs["occupancy_south"],
                inter_obs["occupancy_east"],
                inter_obs["occupancy_west"],
            ])
        return np.array(vector, dtype=np.float32)
    
    def reset(self):
        """Reset environment (gym interface)."""
        obs = self._reset()
        return self._obs_to_vector(obs)
    
    def step(self, action):
        """Step environment (gym interface)."""
        # action is array of 0s/1s (one per intersection)
        response = requests.post(
            f"{self.space_url}/env/step",
            json={"phases": action.tolist()}
        )
        response.raise_for_status()
        obs = response.json()
        
        obs_vector = self._obs_to_vector(obs)
        reward = obs.get("reward", 0.0)
        done = obs.get("done", False)
        info = obs.get("metadata", {})
        
        return obs_vector, reward, done, info
    
    def render(self, mode="human"):
        pass


def train_with_sb3(space_url: str, task: str = "easy"):
    """Train PPO agent on RedGrid."""
    
    # Create environment
    env = RedGridEnv(space_url, task=task)
    
    # Create PPO policy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=20,
        verbose=1,
    )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./ppo_checkpoints/",
        name_prefix="redgrid_ppo"
    )
    
    # Train
    print(f"Training on {task} task...")
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # Save final model
    model.save(f"redgrid_ppo_{task}")
    print(f"Model saved to redgrid_ppo_{task}.zip")
    
    return model


if __name__ == "__main__":
    import sys
    
    space_url = sys.argv[1] if len(sys.argv) > 1 else \
                "https://username-redgrid-environment.hf.space"
    
    print(f"Connecting to {space_url}...")
    train_with_sb3(space_url, task="easy")
```

Run:
```bash
python sb3_train.py https://{username}-{space_name}.hf.space
```

### Option 3: Training Inside Space (CPU)

For lightweight training directly in Space:

1. Create `train_in_space.py` in your repository root
2. Modify `Dockerfile` to run training:

```dockerfile
# ... (previous Dockerfile content)

# Install training dependencies
RUN pip install --no-cache-dir \
    stable-baselines3[extra] \
    gymnasium \
    wandb

# Copy training script
COPY train_in_space.py /app/

# Run server and training in parallel
CMD python -u train_in_space.py
```

---

## Monitor & Debug

### Check Server Logs

```bash
# Via HF Spaces UI
# Go to https://huggingface.co/spaces/{username}/{space_name} → Logs

# Via API
curl https://{username}-{space_name}.hf.space/health
```

### Test Endpoints

```bash
SPACE_URL="https://{username}-{space_name}.hf.space"

# 1. Health check
curl $SPACE_URL/docs -w "\n"

# 2. Reset with seed
curl -X POST $SPACE_URL/env/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}' | jq '.'

# 3. Single step
curl -X POST $SPACE_URL/env/step \
  -H "Content-Type: application/json" \
  -d '{"phases": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}' | jq '.'

# 4. Check response structure
curl -X POST $SPACE_URL/env/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' | jq '.intersections[0]'
```

### Common Issues

| Issue | Solution |
|-------|----------|
| **Port binding error** | Ensure Dockerfile uses port 7860 |
| **Module import error** | Check `COPY redgrid /app/redgrid` in Dockerfile |
| **Timeout during reset** | Increase HF Space timeout; check network connectivity |
| **Agent crashes** | Validate action format: `{"phases": [0, 1, ...]}` |
| **Memory usage high** | Reduce batch size in RL training; use GPU with more VRAM |

### Enable Logging

Modify `server/app.py`:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info(f"Received action: {action}")
logger.info(f"Environment response: {observation}")
```

---

## Cost Optimization

### Hardware Choices

| Hardware | Cost | Best For |
|----------|------|----------|
| **CPU Basic** | $0 | Development, slow training |
| **CPU Upgrade** | $9/month | Baseline agents, inference |
| **GPU T4** | $9/month | Fast RL training (~10x speedup) |
| **GPU A100** | $60/month | Large-scale experiments |

### Recommendations

1. **Development Phase**: Use CPU Basic (free), train locally on powerful machine
2. **Testing**: Use CPU Upgrade with small episodes to validate environment
3. **Full Training**: Duplicate Space on GPU T4, run training overnight
4. **Production Inference**: Use CPU Basic + caching layer

### Optimize Environment Steps

- **Reduce Episode Length**: Use shorter episodes for faster iteration
- **Batch API Calls**: Send multiple steps in parallel requests
- **Use Lighter Networks**: For RL agent (smaller model = faster inference)

Example: Train on easy task instead of hard initially
```python
train_agent(space_url, task="easy", num_episodes=20)  # Fast
# Then progress to harder tasks
```

---

## Advanced: Multi-Agent Training

For training multiple agents simultaneously:

1. **Create separate Spaces** for different difficulty levels
   - `redgrid-easy`
   - `redgrid-medium`
   - `redgrid-hard`

2. **Distributed Training Script**:

```python
import concurrent.futures
from train_agent import train_agent

spaces = {
    "easy": "https://user-redgrid-easy.hf.space",
    "medium": "https://user-redgrid-medium.hf.space",
    "hard": "https://user-redgrid-hard.hf.space",
}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(train_agent, url, task): task 
        for task, url in spaces.items()
    }
    for future in concurrent.futures.as_completed(futures):
        task = futures[future]
        print(f"{task} training complete")
```

---

## Next Steps

1. **Monitor Performance**: Track grader scores during training
2. **Experiment with Algorithms**: Compare PPO vs DQN vs A3C
3. **Hyperparameter Tuning**: Optimize learning rate, network size
4. **Curriculum Learning**: Start with easy → medium → hard
5. **Visualization**: Plot cumulative wait times and rewards over episodes

---

## Support & Links

- **OpenEnv Docs**: https://github.com/huggingface/openenv
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io
- **FastAPI Docs**: https://fastapi.tiangolo.com

