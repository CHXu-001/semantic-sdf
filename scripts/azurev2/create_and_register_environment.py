import argparse
import logging

from azure.ai.ml.entities import BuildContext, Environment
from azure_ml_utils.azure_connect import get_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "environmentname", type=str, help="Name assigned to environment"
    )
    parser.add_argument(
        "dockerbuildcontext", type=str, help="Path to Docker file You want to build"
    )
    parser.add_argument(
        "description",
        type=str,
        help="Description of the environment in few words",
    )
    args = parser.parse_args()

    pipeline_job_env = Environment(
        build=BuildContext(path=args.dockerbuildcontext),
        name=args.environmentname,
        description=args.description,
        tags={},
    )

    ml_client = get_client()
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    logging.info(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, "
        "the environment version is {pipeline_job_env.version}"
    )


if __name__ == "__main__":
    main()
