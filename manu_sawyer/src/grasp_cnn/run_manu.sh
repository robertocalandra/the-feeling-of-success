S=0
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.gel_im_v1($S), 0)" &
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.im_v1($S), 1)" &
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.gel_v1($S), 2)" &
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.depth_v1($S), 3)" &
sleep 20000
killall python ipython
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.gel0_v1($S), 0)" &
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.gel1_v1($S), 1)" &
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.im_ee_v1($S), 2)" &
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.gel_im_depth_v1($S), 3)" &
sleep 20000
killall python ipython
ipython -c "import grasp_net as net, manu_params; net.train(manu_params.press_v1($S), 0)"


