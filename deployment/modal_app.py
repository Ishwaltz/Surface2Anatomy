import modal

# Define Modal application
app = modal.App("surface2anatomy-api")

# Define container image with GPU support and required ML libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libgomp1")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "python-multipart==0.0.9",
        "pydantic==2.7.4",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-learn==1.5.0",
        "trimesh==4.4.1",
        "open3d==0.18.0",
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=120,
    scaledown_window=300,
    mounts=[
        modal.Mount.from_local_dir("api", remote_path="/root/api"),
        modal.Mount.from_local_dir("sharon", remote_path="/root/sharon"),
        modal.Mount.from_local_dir("tools", remote_path="/root/tools"),
        modal.Mount.from_local_dir("experiments/phase10R/checkpoints", remote_path="/root/experiments/phase10R/checkpoints"),
        modal.Mount.from_local_dir("experiments/phase16_brain/checkpoints", remote_path="/root/experiments/phase16_brain/checkpoints"),
    ],
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from api.main import app as fastapi_app_instance
    return fastapi_app_instance
