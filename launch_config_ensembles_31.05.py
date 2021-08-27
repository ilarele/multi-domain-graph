import configparser
import os
import sys
from datetime import datetime

import numpy as np

os.system("mkdir -p generated_configs/")

# dst domain
domain_id = sys.argv[1]
# base template
cfg_template = sys.argv[2]
# hypersim_v2_template_eval_iter1_train_26_30_31_32.ini
# hypersim_v2_template_eval_iter1_train_12_13_14_15.ini

# intro
cfg_out = "generated_configs/launch_ensembles_%s_%s_%s.ini" % (
    domain_id, str(datetime.now()), cfg_template[:-4])
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_type", '2')
config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)
tensorboard_prefix = config.get("Logs", "tensorboard_prefix")

#epochs = np.array([200, 150, 100, 50])
epochs = np.array([100])  #200, 150, 100, 50])

for epoch_idx in epochs:

    # run simple mean and median
    if True:
        ## test simple mean
        config.set('Ensemble', 'enable_simple_mean', 'yes')
        tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
        config.set(
            "Logs", "tensorboard_prefix", "%s_%s_%s_simple_mean" %
            (tensorboard_prefix, str(epoch_idx), domain_id))
        config.set('Edge Models', 'start_epoch', str(epoch_idx))
        with open(cfg_out, "w") as fd:
            config.write(fd)

        os.system('python main.py "%s"' % (cfg_out))

        ## test simple median
        config.set('Ensemble', 'enable_simple_mean', 'no')
        config.set('Ensemble', 'enable_simple_median', 'yes')
        config.set(
            "Logs", "tensorboard_prefix", "%s_%s_%s_simple_median" %
            (tensorboard_prefix, str(epoch_idx), domain_id))
        config.set('Edge Models', 'start_epoch', str(epoch_idx))
        with open(cfg_out, "w") as fd:
            config.write(fd)

        os.system('python main.py "%s"' % (cfg_out))

    # run configs
    if True:
        ## test metrics without variance
        config.set('Ensemble', 'enable_simple_mean', 'no')
        config.set('Ensemble', 'enable_simple_median', 'no')
        config.set('Ensemble', 'fix_variance', 'no')
        config.set('Edge Models', 'start_epoch', str(epoch_idx))
        #for sim_f in ['l1', 'l2', 'ssim', 'lpips', 'dist_mean', 'psnr']:
        for sim_f in ['lpips_per_channel', 'l1', 'ssim']:
            config.set('Ensemble', 'similarity_fct', sim_f)
            for kernel_f in ['flat_weighted', 'flat', 'gauss']:
                config.set('Ensemble', 'kernel_fct', kernel_f)
                ths = np.array([0.25, 0.5, 0.75, 1])
                for th in ths:
                    config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                    for comb_type in ['mean', 'median']:
                        config.set('Ensemble', 'comb_type', comb_type)
                        config.set(
                            "Logs", "tensorboard_prefix",
                            "%s_%s_%s__%s__%s__%s__%s" %
                            (tensorboard_prefix, str(epoch_idx), domain_id,
                             sim_f, kernel_f, comb_type, str(th)))
                        with open(cfg_out, "w") as fd:
                            config.write(fd)

                        os.system('python main.py "%s"' % (cfg_out))