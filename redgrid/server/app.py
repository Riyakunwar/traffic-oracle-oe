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
except (ImportError, ModuleNotFoundError):
    from models import TrafficAction, TrafficObservation
    from server.traffic_environment import TrafficEnvironment


app = create_app(
    TrafficEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="redgrid",
    max_concurrent_envs=1,
)


def main():
    """Run the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
