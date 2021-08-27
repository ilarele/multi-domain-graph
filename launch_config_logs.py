import configparser
import os
import sys
import numpy as np
from datetime import datetime

os.system("mkdir -p generated_configs/")

domain_id = sys.argv[1]
cfg_template = sys.argv[2]
# replica_template_iter1_logs.ini
epoch_idx = np.int32(sys.argv[3])

# intro
cfg_out = "generated_configs/launch_%s_%s.ini" % (domain_id, str(
    datetime.now()))
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)

# SET model type
if domain_id in [
        "edges_dexined", "sobel_large", "sobel_small", "sobel_medium", "hsv"
]:
    config.set("Edge Models", "regression_losses", "l2")
    config.set("Edge Models", "regression_losses_weights", "1")
orig_tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
config.set('Edge Models', 'start_epoch', str(epoch_idx))
config.set("Logs", "tensorboard_prefix",
           "%s_%s_%d" % (orig_tensorboard_prefix, domain_id, epoch_idx))

with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
