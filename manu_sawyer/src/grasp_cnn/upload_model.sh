# cd /home/ao/Dropbox/gelsight-grasp/results/press-data-v6
# rm model.zip
# zip model.zip training/net.tf-2600*
# scp model.zip owens@knodel:public_html/press-model/model.zip

# cd /home/ao/Dropbox/gelsight-grasp/results/press-data-v8
# rm model.zip
# #zip model.zip training/net.tf-2600*
# zip model.zip training/net.tf-2200*
# scp model.zip owens@knodel:public_html/press-model/model.zip


# cd /home/ao/Dropbox/gelsight-grasp/results/press-data-v11
# rm model.zip
# zip model.zip training/net.tf-4600*
# scp model.zip owens@knodel:public_html/press-model/model.zip



# cd ../results/grasp-params/gel-v1/
# rm grasp_model.zip
# zip grasp_model.zip training/net.tf-2000*
# scp grasp_model.zip owens@knodel:public_html/press-model/grasp_model.zip



cd ../results/grasp-params/gel-v4/
rm grasp_model.zip
zip grasp_model.zip training/net.tf-3999*
scp grasp_model.zip owens@knodel:public_html/press-model/grasp_model.zip


