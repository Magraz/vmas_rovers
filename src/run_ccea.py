from learning.ccea import runCCEA
import os

import argparse

if __name__ == "__main__":

    # Arg parser variables
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpc", default=False, help="use hpc config files", action="store_true"
    )
    parser.add_argument(
        "--batch",
        default="",
        help="Experiment batch",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="",
        help="Experiment name",
        type=str,
    )
    parser.add_argument("--trial_id", default=0, help="Sets trial ID", type=int)

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if args["hpc"]:
        dir_path = "/nfs/stak/users/agrazvam/hpc-share/vmas_rovers/src"

    # Set configuration folder
    batch_dir = os.path.join(dir_path, "experiments", "yamls", args["batch"])

    # Run learning algorithm
    runCCEA(
        batch_dir=batch_dir,
        batch_name=args["batch"],
        experiment_name=args["name"],
        trial_id=args["trial_id"],
    )
