nohup python launch_config_save.py rgb hypersim_v2_template_save_iter1_v2.ini > logs_hypersim_v2_save_iter1_v2_rgb.out &
wait
nohup python launch_config_save.py depth_n_1_xtc hypersim_v2_template_save_iter1_v2.ini > logs_hypersim_v2_save_iter1_v2_depth_n_1_xtc.out &
wait 
nohup python launch_config_save.py normals_xtc hypersim_v2_template_save_iter1_v2.ini > logs_hypersim_v2_save_iter1_v2_normals_xtc.out &
