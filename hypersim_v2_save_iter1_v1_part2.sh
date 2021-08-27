nohup python launch_config_save.py hsv hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_hsv.out &
wait
nohup python launch_config_save.py cartoon_wb hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_cartoon_wb.out &
wait 
nohup python launch_config_save.py sobel_large hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_sobel_large.out &
wait 
nohup python launch_config_save.py sobel_medium hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_sobel_medium.out &
wait 
nohup python launch_config_save.py sobel_small hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_sobel_small.out &
wait 
nohup python launch_config_save.py superpixel_fcn hypersim_v2_template_save_iter1_v1.ini > logs_hypersim_v2_save_iter1_v1_superpixel_fcn.out &