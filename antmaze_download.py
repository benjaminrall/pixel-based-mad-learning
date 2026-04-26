import argparse
import ogbench
import numpy as np
from numpy.typing import NDArray
import os
import zipfile
import minari
import shutil
from minari.data_collector.episode_buffer import EpisodeBuffer
import gymnasium as gym
from tqdm import tqdm

def load_npz_mmap(npz_path: str, cache_dir: str | None = None) -> dict[str, NDArray]:
    """Extracts an NPZ archive and returns memory mapped arrays of its contents."""

    if cache_dir is None:
        cache_dir = npz_path.removesuffix('.npz')
    os.makedirs(cache_dir, exist_ok=True)

    mmap_dict = {}

    with zipfile.ZipFile(npz_path, 'r') as zf:
        files = zf.namelist()

        for f in files:
            if not f.endswith('.npy'):
                continue

            array_name = f.removesuffix('.npy')

            extract_path = os.path.join(cache_dir, f)

            if not os.path.exists(extract_path):
                with zf.open(f) as source, open(extract_path, 'wb') as target:
                    shutil.copyfileobj(source, target)

            mmap_dict[array_name] = np.load(extract_path, mmap_mode='r')

    return mmap_dict


FLUSH_CHUNK_SIZE = 100

if __name__ == '__main__':
    # Sets up command line argument parsing
    parser = argparse.ArgumentParser(description="Downloads OGBench AntMaze datasets and converts them to local Minari dataset format.")
    parser.add_argument('-s', '--source', type=str, required=True, 
                        help="OGBench dataset to download (e.g., visual-antmaze-medium-navigate-v0)")
    parser.add_argument('-t', '--target', type=str, required=True, 
                        help="Target Minari dataset identifier (e.g., ogbench/antmaze/visual-medium-navigation-v0)")
    
    args = parser.parse_args()
    
    dataset_id = args.source
    minari_id = args.target

    # Automatic inference of observation space based on whether the environment is visual or not
    if 'visual' in dataset_id.lower():
        base_obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    else:
        base_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
        
    obs_space = gym.spaces.Dict({
        'observation': base_obs_space,
        'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32) 
    })
    
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    # Downloading and extracting data from the OGBench dataset
    print(f"Downloading OGBench dataset: {dataset_id}")
    ogbench.download_datasets([dataset_id], dataset_dir='.cache/ogbench/')

    print("Loading OGBench data via memory mapping...")
    ogbench_data = load_npz_mmap(f'.cache/ogbench/{dataset_id}.npz')

    obs = ogbench_data['observations']
    terminals = ogbench_data['terminals']
    actions = ogbench_data['actions']
    pos = ogbench_data['qpos'][:, :2]
    ep_ends = np.where(terminals == 1)[0]

    total_steps = len(obs)

    # Creates new Minari dataset locally
    print(f"Creating Minari dataset: {minari_id}")
    dataset = minari.create_dataset_from_buffers(
        dataset_id=minari_id,
        buffer=[],
        observation_space=obs_space,
        action_space=action_space,
    )

    current_buffers = []
    start_idx = 0

    # Writes episode data from the OGBench dataset to the Minari dataset 
    for end_idx in tqdm(ep_ends):
        episode_length = end_idx - start_idx + 1

        ep_obs = {
            'observation': obs[start_idx:end_idx+1],
            'achieved_goal': pos[start_idx:end_idx+1],
        }

        ep_buffer = EpisodeBuffer(
            observations=ep_obs,
            actions=list(actions[start_idx:end_idx]),
            rewards=list(np.zeros(episode_length - 1, dtype=np.float32)),
            terminations=list(terminals[start_idx:end_idx]),
            truncations=list(np.zeros(episode_length - 1, dtype=bool)),
            infos={}
        )

        current_buffers.append(ep_buffer)

        if len(current_buffers) >= FLUSH_CHUNK_SIZE:
            dataset.update_dataset_from_buffer(current_buffers)
            current_buffers = []

        start_idx = end_idx + 1
    
    if current_buffers:
        dataset.update_dataset_from_buffer(current_buffers)

    print("Processing complete.")