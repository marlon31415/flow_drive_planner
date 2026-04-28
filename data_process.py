import os
import argparse
import json
from pathlib import Path

from flow_drive.data_process.data_processor import DataProcessor
from flow_drive.utils.train_utils import load_params

import interplan
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)


def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):

    scenario_types = None

    scenario_tokens  # List of scenario tokens to include
    log_names = log_names  # Filter scenarios by log names
    map_names = None  # Filter scenarios by map names

    num_scenarios_per_type  # Number of scenarios per type
    limit_total_scenarios  # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None  # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None  # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True  # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = (
        False  # Whether to remove scenarios where the mission goal is invalid
    )
    shuffle  # Whether to shuffle the scenarios

    ego_start_speed_threshold = (
        None  # Limit to scenarios where the ego reaches a certain speed from below
    )
    ego_stop_speed_threshold = (
        None  # Limit to scenarios where the ego reaches a certain speed from above
    )
    speed_noise_tolerance = None  # Value at or below which a speed change between two timepoints should be ignored as noise.

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')
    parser.add_argument('--split', type=str, default='train', help='data split to process (train, val, val14, interplan)')
    args = parser.parse_args()
    # fmt: on

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    params = load_params(
        os.path.join(os.path.dirname(__file__), "flow_drive", "config", "config.yaml")
    )
    params.data_processing.save_path = args.save_path
    if args.split == "interplan":
        params.data_processing.interplan = True
    processor = DataProcessor(params.data_processing)

    sensor_root = None
    db_files = None
    log_names = None

    # Preprocess the training, validation or test data
    if args.split == "train":
        scenarios = "nuplan_train.json"
    elif args.split == "interplan":
        scenarios = "nuplan_test.json"
    else:  # "val" or "val14"
        scenarios = "nuplan_val.json"
    with open(
        os.path.join(os.path.dirname(__file__), scenarios), "r", encoding="utf-8"
    ) as file:
        log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(
        args.data_path, args.map_path, sensor_root, db_files, map_version
    )
    if args.split == "val14":
        val14_params = load_params(
            os.path.join(
                os.path.dirname(__file__),
                "flow_drive",
                "config",
                "scenario_filter",
                "val14.yaml",
            )
        )
        val14_tokens = val14_params.scenario_tokens.to_list()
        scenario_filter = ScenarioFilter(
            *get_filter_parameters(
                args.scenarios_per_type,
                args.total_scenarios,
                args.shuffle_scenarios,
                log_names=None,
                scenario_tokens=val14_tokens,
            )
        )
    elif args.split == "interplan":
        interplan_params = load_params(
            os.path.join(
                Path(
                    interplan.__path__[0],
                    "planning",
                    "script",
                    "config",
                    "common",
                    "scenario_filter",
                    "benchmark_scenarios.yaml",
                )
            )
        )
        interplan_benchmark_tokens = interplan_params.scenario_tokens.to_list()
        # Remove descriptors (e.g., "-s0", "-lg") and duplicates
        interplan_benchmark_tokens = list(
            set([token.split("-")[0] for token in interplan_benchmark_tokens])
        )
        interplan_modifications = load_params(
            os.path.join(
                Path(
                    interplan.__path__[0],
                    "planning",
                    "script",
                    "config",
                    "common",
                    "scenario_filter",
                    "modifications",
                    "interPlan_modifications.yaml",
                )
            )
        )
        interplan_mod_details = interplan_modifications[
            "modification_details_dictionary"
        ]
        interplan_mod_tokens = list(interplan_mod_details.keys())
        interplan_tokens = list(set(interplan_benchmark_tokens + interplan_mod_tokens))
        scenario_filter = ScenarioFilter(
            *get_filter_parameters(
                args.scenarios_per_type,
                args.total_scenarios,
                args.shuffle_scenarios,
                log_names=log_names,
                scenario_tokens=interplan_tokens,
            )
        )
    else:
        scenario_filter = ScenarioFilter(
            *get_filter_parameters(
                args.scenarios_per_type,
                args.total_scenarios,
                args.shuffle_scenarios,
                log_names=log_names,
            )
        )

    print("Loading scenarios...")
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter

    # processor.work(scenarios)
    processor.work_multiprocess(scenarios, n_processes=64)

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith(".npz")]

    # Save the list to a JSON file
    if args.split == "train":
        json_list_path = os.path.join(args.save_path, "scenarios_training.json")
    elif args.split == "val":
        json_list_path = os.path.join(args.save_path, "scenarios_validation.json")
    elif args.split == "val14":
        json_list_path = os.path.join(args.save_path, "scenarios_val14.json")
    elif args.split == "interplan":
        json_list_path = os.path.join(args.save_path, "scenarios_interplan.json")
    with open(json_list_path, "w") as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names")
