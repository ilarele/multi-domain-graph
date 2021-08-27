import configparser
import os
import sys
import numpy as np
from datetime import datetime

os.system("mkdir -p generated_configs/")

domain_id = sys.argv[1]
cfg_template = sys.argv[2]
# replica_template_iter1.ini
# replica_template_train_iter2.ini
# hypersim_template_train_iter1.ini
# hypersim_v2_template_iter1.ini
# 0. RGBModel(full_experts),
# 1. DepthModelXTC(full_experts),
# 2. SurfaceNormalsXTC(full_experts),
# 3. EdgesModel(full_experts),
# 4. HalftoneModel(full_experts, 0),
# 5. SSegHRNet
# 6. Grayscale(full_experts),
# 7. HSVExpert(full_experts),
# 8. CartoonWB(full_experts),
# 9. SobelEdgesExpertSigmaLarge(full_experts),
# 10. SobelEdgesExpertSigmaMedium(full_experts),
# 11. SobelEdgesExpertSigmaSmall(full_experts),
# 12. SuperPixel(full_experts),
#selector_map = 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12
#selector_map = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

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

#config.set("Experts", "selector_map", "2, 12, 11")  # mdg-7
#config.set("Experts", "selector_map", "2, 9, 10")  # mdg-5
#config.set("Experts", "selector_map", "2, 7, 8")  # mdg-3
#config.set("Experts", "selector_map", "0, 12, 11")  # mdg-9
#config.set("Experts", "selector_map", "0, 9, 10")  # mdg-11
#config.set("Experts", "selector_map", "0, 8")  # mdg-5

# iter2 - v2 - twd depth
#config.set("Experts", "selector_map", "1, 0, 2, 3")  # mdg
#config.set("Experts", "selector_map", "1, 4, 5, 6")  # mdg-2
#config.set("Experts", "selector_map", "1, 7, 8, 9")  # mdg-3
#config.set("Experts", "selector_map", "1, 10, 11, 12")  # mdg-4
#config.set("Experts", "selector_map", "1, 12")  # mdg-5

# iter2 - v2 - twd normals
#config.set("Experts", "selector_map", "2, 0, 1, 3")  # mdg-7
#config.set("Experts", "selector_map", "2, 4, 5, 6")  # mdg-9
#config.set("Experts", "selector_map", "2, 7, 8, 9")  # mdg-5
#config.set("Experts", "selector_map", "2, 10, 11, 12")  # mdg-10

# iter2 - v2 - twd rgb
#config.set("Experts", "selector_map", "0, 1, 2, 3")  # mdg-12
#config.set("Experts", "selector_map", "0, 4, 5, 6")  # mdg-6
#config.set("Experts", "selector_map", "0, 7, 8, 9")  # mdg-8
#config.set("Experts", "selector_map", "0, 10, 11, 12")  #mdg-11
#config.set("Experts", "selector_map", "0, 12")  #mdg-3
#config.set("Experts", "selector_map", "0, 6")  #mdg-9

# iter2 - v3 - twd depth
#config.set("Experts", "selector_map", "1, 0, 2, 3")  # mdg
#config.set("Experts", "selector_map", "1, 4, 5, 6")  # mdg-2
#config.set("Experts", "selector_map", "1, 7, 8, 9")  # mdg-3
#config.set("Experts", "selector_map", "1, 10, 11, 12")  # mdg-4
#config.set("Experts", "selector_map", "1, 3, 6")  # mdg
#config.set("Experts", "selector_map", "1, 9, 11, 12")  # mdg-2
#config.set("Experts", "selector_map", "1, 12")  # mdg

# iter2 - v3 - twd normals
#config.set("Experts", "selector_map", "2, 0, 1, 3")  # mdg-5
#config.set("Experts", "selector_map", "2, 4, 5, 6")  # mdg-6
#config.set("Experts", "selector_map", "2, 7, 8, 9")  # mdg-7
#config.set("Experts", "selector_map", "2, 10, 11, 12")  # mdg-9
#config.set("Experts", "selector_map", "2, 5, 6")  # mdg-3
#config.set("Experts", "selector_map", "2, 8, 9, 12")  # mdg-4
#config.set("Experts", "selector_map", "2, 12")  # mdg-3

# iter2 - v3 - twd rgb
#config.set("Experts", "selector_map", "0, 1, 2, 3")  # mdg-8
#config.set("Experts", "selector_map", "0, 4, 5, 6")  # mdg-10
#config.set("Experts", "selector_map", "0, 7, 8, 9")  #mdg-5
#config.set("Experts", "selector_map", "0, 10, 11, 12")  #mdg-6
config.set("Experts", "selector_map", "0, 12")  # mdg-3

#config.set("Edge Models", "model_type", "1")

tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
config.set("Logs", "tensorboard_prefix",
           "%s_%s" % (tensorboard_prefix, domain_id))

with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
