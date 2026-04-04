"""FastAPI application for the Traffic Signal OpenEnv environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install 'openenv[core]>=0.2.0'"
    ) from e

try:
    from ..models import TrafficAction, TrafficObservation
    from .traffic_environment import TrafficEnvironment
except ModuleNotFoundError:
    from models import TrafficAction, TrafficObservation
    from server.traffic_environment import TrafficEnvironment


app = create_app(
    TrafficEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="redgrid",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the server directly."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
