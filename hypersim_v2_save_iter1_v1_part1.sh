nohup python launch_config_save.py rgb hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_rgb.out &
wait
nohup python launch_config_save.py depth_n_1_xtc hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_depth_n_1_xtc.out &
wait 
nohup python launch_config_save.py edges_dexined hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_edges_dexined.out &
wait 
nohup python launch_config_save.py halftone_gray hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_halftone_gray.out &
wait 
nohup python launch_config_save.py sem_seg_hrnet hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_sem_seg_hrnet.out &
wait 
nohup python launch_config_save.py grayscale hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_grayscale.out &
