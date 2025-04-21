import os
import logging
import datetime
from pathlib import Path


def get_method_name(args):
    method_name = args.model.split("/")[-1] + "_" + args.method
    if args.method in ['snapkv', 'snapkv2']:
        method_name += f"_w{args.window_size}_k{args.kernel_size}_cr{args.compression_ratio}"
    elif args.method == 'balancekv':
        method_name += f"_itr{args.itrs}_g{args.gamma}_t{args.temp}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    else:
        pass
    return method_name


def set_logger(args, dataset_name, method_name, task="longbench_v1"):
    save_filename = ""
    handlers = [logging.StreamHandler()]
    
    if not args.debug:
        task_path = Path(f"./{args.prefix}results/") / f"{args.dataset}"
        datestr = datetime.datetime.now().strftime('%y%m%d%H%M')
        result_dir = task_path / method_name
        result_dir.mkdir(exist_ok=True, parents=True)
        output_file = f"{args.datadir.replace('_e','')}_"+\
            (f"_fraction{args.fraction}" if args.fraction < 1.0 else "")+\
            f"{datestr}"
        save_filename = result_dir / f"{output_file}.jsonl"
        
        log_path = Path(f"./{args.prefix}logs") / f"{args.dataset}" / method_name
        log_path.mkdir(exist_ok=True, parents=True)
        log_path = log_path / f"{output_file}.txt"
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=logging.INFO, 
        format=f'[%(asctime)s]{dataset_name}|{args.method}| %(message)s',
        datefmt='%y%m%d %H:%M:%S',
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    return logger, save_filename


def reset_logger():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True