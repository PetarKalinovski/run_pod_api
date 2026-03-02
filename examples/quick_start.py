"""Quick start example for runpodx.

Set RUNPOD_API_KEY in your environment before running:
    export RUNPOD_API_KEY=rp_xxxxxx
    uv run python examples/quick_start.py
"""

from runpodx import RunPodClient

client = RunPodClient()

# --- List existing pods ---
print("Current pods:")
for pod in client.pods.list():
    print(f"  {pod.name} ({pod.id}) - {pod.desired_status} - ${pod.cost_per_hr}/hr")

# --- Create a pod from TOML ---
# pod = client.pods.create("examples/basic_pod.toml")
# print(f"\nCreated: {pod.id}")
# pod.wait_until_ready(timeout=120)
# print(f"Running at {pod.public_ip}")

# --- Or create programmatically ---
# from runpodx import PodCreateConfig, GpuConfig
#
# config = PodCreateConfig(
#     name="my-experiment",
#     image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
#     gpu=GpuConfig(count=2, type_ids=["NVIDIA GeForce RTX 4090"]),
#     env={"WANDB_PROJECT": "my-experiment"},
# )
# pod = client.pods.create(config)
# pod.wait_until_ready()
# print(pod)

# --- Stop when done ---
# pod.stop()

client.close()
