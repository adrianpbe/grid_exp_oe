
TAGGED_MINIGRID_ENVS = [
    {
        "name": "BlockedUnlockPickup",
        "configs": ["MiniGrid-BlockedUnlockPickup-v0"],
        "tags": ["solvable_without_language"]
    },
    {
        "name": "LavaCrossing",
        "configs": ["MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-LavaCrossingS9N2-v0", "MiniGrid-LavaCrossingS9N3-v0", "MiniGrid-LavaCrossingS11N5-v0"],
        "tags": ["solvable_without_language", "safety", "suitable_for_curriculum"]
    },
    {
        "name": "SimpleCrossing",
        "configs": ["MiniGrid-SimpleCrossingS9N1-v0", "MiniGrid-SimpleCrossingS9N2-v0", "MiniGrid-SimpleCrossingS9N3-v0", "MiniGrid-SimpleCrossingS11N5-v0"],
        "tags": ["solvable_without_language", "suitable_for_curriculum"]
    },
    {
        "name": "DistShift",
        "configs": ["MiniGrid-DistShift1-v0", "MiniGrid-DistShift2-v0"],
        "tags": ["solvable_without_language"]
    },
    {
        "name": "DoorKey",
        "configs": ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0", "MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-16x16-v0"],
        "tags": ["solvable_without_language", "hard_sparse", "suitable_for_curriculum"]
    },
    {
        "name": "Fetch",
        "configs": ["MiniGrid-Fetch-5x5-N2-v0", "MiniGrid-Fetch-6x6-N2-v0", "MiniGrid-Fetch-8x8-N3-v0"],
        "tags": ["requires_language"]
    },
    {
        "name": "FourRooms",
        "configs": ["MiniGrid-FourRooms-v0"],
        "tags": ["solvable_without_language"]
    },
    {
        "name": "GoToDoor",
        "configs": ["MiniGrid-GoToDoor-5x5-v0", "MiniGrid-GoToDoor-6x6-v0", "MiniGrid-GoToDoor-8x8-v0"],
        "tags": ["requires_language"]
    },
    {
        "name": "GoToObject",
        "configs": ["MiniGrid-GoToObject-6x6-N2-v0", "MiniGrid-GoToObject-8x8-N2-v0"],
        "tags": ["requires_language"]
    },
    {
        "name": "KeyCorridor",
        "configs": ["MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-KeyCorridorS3R2-v0", "MiniGrid-KeyCorridorS3R3-v0", "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-KeyCorridorS5R3-v0", "MiniGrid-KeyCorridorS6R3-v0"],
        "tags": ["solvable_without_language", "suitable_for_curriculum"]
    },
    {
        "name": "LavaGap",
        "configs": ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaGapS6-v0", "MiniGrid-LavaGapS7-v0"],
        "tags": ["solvable_without_language", "safety", "suitable_for_curriculum"]
    },
    {
        "name": "LockedRoom",
        "configs": ["MiniGrid-LockedRoom-v0"],
        "tags": ["hard_sparse"]
    },
    {
        "name": "Memory",
        "configs": ["MiniGrid-MemoryS17Random-v0", "MiniGrid-MemoryS13Random-v0", "MiniGrid-MemoryS13-v0", "MiniGrid-MemoryS11-v0"],
        "tags": ["solvable_without_language", "requires_memory"]
    },
    {
        "name": "MultiRoom",
        "configs": ["MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v0", "MiniGrid-MultiRoom-N6-v0"],
        "tags": ["solvable_without_language", "hard_sparse", "suitable_for_curriculum"]
    },
    {
        "name": "ObstructedMaze",
        "configs": ["MiniGrid-ObstructedMaze-1Dlhb-v0"],
        "tags": ["solvable_without_language"]
    },

    {
        "name": "ObstructedMaze_Full_V1",
        "configs": ["MiniGrid-ObstructedMaze-Full-v1"],
        "tags": ["solvable_without_language"]
    },
    {
        "name": "PutNear",
        "configs": ["MiniGrid-PutNear-6x6-N2-v0", "MiniGrid-PutNear-8x8-N3-v0"],
        "tags": ["requires_language"],
    },
    {
        "name": "RedBlueDoors",
        "configs": ["MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-8x8-v0"],
        "tags": ["solvable_without_language"]
    },
    {
        "name": "Unlock",
        "configs": ["MiniGrid-Unlock-v0"],
        "tags": ["solvable_without_language"]
    },
    {
        "name": "UnlockPickup",
        "configs": ["MiniGrid-UnlockPickup-v0"],
        "tags": ["solvable_without_language"]
    }
]


if __name__ == "__main__":
    # Iterates over all envs, check that is solvable without language and prints the name
    #  as well as the list of configs
    for env in TAGGED_MINIGRID_ENVS:
        if "solvable_without_language" in env["tags"]:
            print(env["name"])
            print(env["configs"])
    from grid_exp_oe.env import create_env
    # Test that all models are instantiated and validated by the create_env function
    # and that one (random) action is taken
    for env in TAGGED_MINIGRID_ENVS:
        for config in env["configs"]:
            env = create_env(config)
            _ = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
    print("All environments are correctly instantiated")
