#!/usr/bin/env python3
"""
Surface2Anatomy Hugging Face Space One-Click Deployment Script
Uploads the complete hf_space/ package including model weights to Hugging Face Spaces.
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def deploy(repo_id: str, token: str = None, space_sdk: str = "gradio"):
    print(f"🚀 Deploying Surface2Anatomy backend to Hugging Face Space: {repo_id}...")
    api = HfApi(token=token)
    
    # Verify token
    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as HF User: {user_info.get('name', user_info.get('preferred_username', 'Unknown'))}")
    except Exception as e:
        print(f"❌ HF Authentication failed: {e}")
        print("Please supply a valid Write token from https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Create Space if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=space_sdk,
            exist_ok=True,
            token=token
        )
        print(f"✅ Space repository verified: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"⚠️ Notice when creating repo: {e}")

    # Upload folder
    hf_folder = Path(__file__).resolve().parent / "hf_space"
    print(f"📦 Uploading files and neural network weights from {hf_folder}...")
    
    api.upload_folder(
        folder_path=str(hf_folder),
        repo_id=repo_id,
        repo_type="space",
        token=token
    )
    
    space_subdomain = repo_id.replace("/", "-").lower()
    live_url = f"https://{space_subdomain}.hf.space"
    print("\n🎉 Deployment successfully triggered!")
    print(f"🌐 Space Dashboard: https://huggingface.co/spaces/{repo_id}")
    print(f"⚡ Live API URL: {live_url}")
    print(f"🏥 Health Check: {live_url}/health")
    print(f"📖 Docs: {live_url}/docs")
    print("\nTo link this to your Vercel frontend:")
    print(f"npx vercel env add NEXT_PUBLIC_INFERENCE_API_URL production")
    print(f"Value: {live_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Surface2Anatomy to HF Spaces")
    parser.add_argument("--repo", required=True, help="HF repo name in format username/space-name")
    parser.add_argument("--token", default=None, help="Hugging Face Write Token (or uses stored token)")
    args = parser.parse_args()
    deploy(repo_id=args.repo, token=args.token)
