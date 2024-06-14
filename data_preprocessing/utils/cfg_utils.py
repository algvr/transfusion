import logging
import sys

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import get_cfg
from detectron2.modeling.meta_arch.build import build_model


def unidet_from_cfg(cfg, evaluate=True):
    model = build_model(cfg)
    if evaluate:
        model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    # logging.info(f"built model from {cfg} with evaluate:{evaluate}")

    return model


def setup_logger(path=None):
    logger = logging.getLogger(__name__)
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(filename)s:%(lineno)s - %(funcName)20s()]"
    datefmt = "%m/%d/%Y, %H:%M:%S"
    if path != None:
        log_file = path.joinpath("run.log")
        logging.basicConfig(
            level=logging.INFO,
            # filename=log_file,
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
            format=format,
            datefmt=datefmt,
        )
    else:
        logging.basicConfig(level=logging.INFO, format=format, datefmt=datefmt)


def my_setup_cfg(config_file, opts=[], conf=0.4):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_unidet_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
    cfg.freeze()
    return cfg
