"""
Submit the learned-dynamics training job to Azure ML and download results
to results/learned_models/ when complete.

Prerequisites:
    pip install azure-ai-ml azure-identity

Environment variables (add to .env):
    AZURE_ML_SUBSCRIPTION_ID
    AZURE_ML_RESOURCE_GROUP
    AZURE_ML_WORKSPACE
    AZURE_ML_COMPUTE          # name of an existing Compute Cluster (min_nodes=0)
    AZURE_ML_STORAGE_ACCOUNT  # storage account name for the dataset blob container
    AZURE_ML_DATASET_CONTAINER # blob container holding results/dataset/ (default: pendulum)

Usage:
    python scripts/azure_ml_job.py [--epochs 500] [--wait]
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        sys.exit(f"Missing required environment variable: {key}")
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int, default=500)
    ap.add_argument("--hidden",     type=int, default=256)
    ap.add_argument("--layers",     type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--models",     nargs="*", default=["neural_ode", "hnn", "lnn"])
    ap.add_argument("--wait",       action="store_true",
                    help="Block until the job completes then download weights locally")
    ap.add_argument("--local-output", default="results/learned_models",
                    help="Where to save downloaded weights (only used with --wait)")
    args = ap.parse_args()

    try:
        from azure.ai.ml import MLClient, command, Input, Output
        from azure.ai.ml.constants import AssetTypes, InputOutputModes
        from azure.identity import DefaultAzureCredential
    except ImportError:
        sys.exit(
            "azure-ai-ml and azure-identity are required.\n"
            "Run: pip install azure-ai-ml azure-identity"
        )

    subscription_id  = _require_env("AZURE_ML_SUBSCRIPTION_ID")
    resource_group   = _require_env("AZURE_ML_RESOURCE_GROUP")
    workspace        = _require_env("AZURE_ML_WORKSPACE")
    compute_name     = _require_env("AZURE_ML_COMPUTE")
    storage_account  = _require_env("AZURE_ML_STORAGE_ACCOUNT")
    dataset_container = os.getenv("AZURE_ML_DATASET_CONTAINER", "pendulum")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    dataset_uri = (
        f"azureml://subscriptions/{subscription_id}"
        f"/resourcegroups/{resource_group}"
        f"/workspaces/{workspace}"
        f"/datastores/workspaceblobstore"
        f"/paths/{dataset_container}/dataset/"
    )

    job = command(
        display_name="pendulum-learned-dynamics-training",
        experiment_name="double-pendulum-benchmark",
        compute=compute_name,
        environment="azureml:AzureML-pytorch-2.0-ubuntu20.04-py38-cuda11-gpu@latest",
        command=(
            "python scripts/train_learned.py"
            " --dataset-dir ${{inputs.dataset}}"
            " --output-dir  ${{outputs.weights}}"
            f" --epochs {args.epochs}"
            f" --hidden {args.hidden}"
            f" --layers {args.layers}"
            f" --batch-size {args.batch_size}"
            f" --lr {args.lr}"
            f" --models {' '.join(args.models)}"
        ),
        inputs={
            "dataset": Input(
                type=AssetTypes.URI_FOLDER,
                path=dataset_uri,
                mode=InputOutputModes.RO_MOUNT,
            )
        },
        outputs={
            "weights": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            )
        },
        # Upload the full project so train_learned.py + bench/ are available
        code=".",
    )

    submitted = ml_client.jobs.create_or_update(job)
    print(f"Submitted job: {submitted.name}")
    print(f"Studio URL:    {submitted.studio_url}")

    if not args.wait:
        print(
            "\nRun with --wait to block until complete and auto-download weights.\n"
            "Or download manually once the job finishes:\n"
            f"  az ml job download --name {submitted.name} --output-name weights "
            f"--download-path {args.local_output}"
        )
        return

    print("Waiting for job to complete (cluster auto-scales to 0 after) …")
    ml_client.jobs.stream(submitted.name)

    # Download trained weights to local results/learned_models/
    os.makedirs(args.local_output, exist_ok=True)
    ml_client.jobs.download(
        name=submitted.name,
        output_name="weights",
        download_path=args.local_output,
    )
    print(f"\nWeights downloaded to: {args.local_output}/")
    print("You can now run the eval:\n  python scripts/run_eval.py --models neural-ode hnn lnn")


if __name__ == "__main__":
    main()
