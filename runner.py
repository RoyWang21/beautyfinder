if __name__ == "__main__":
    # main file to parse args and execute
    from pathlib import Path
    from config import config
    from bfinder import train, utils
    import argparse

    args_fp = Path(config.CONFIG_DIR, "args.json")
    args = argparse.Namespace(**utils.load_dict(filepath=args_fp))
    data_path = config.DATA_DIR
    print(args)
    print(data_path)
