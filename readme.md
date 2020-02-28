## Jüri Ratas bot

Bot made of Jüri Ratas interviews and talks in Riigikogu. Data used for training model is not comprehensive.
Goal of this project is to have some fun and see if sequnece2sequence model could give human like answers.

### Train model
To train model use notebook 0.1_prepare_jyri_data.ipynb to prepare data and notebook 0.2_train_jyri.ipynb to train model.
Use GPU to seed up training. 

### Use bot
To use bot have model files in models sub folder. Use build_dev.bat (you need to convert it.sh files if using linux) to build docker image.
Use run_dev.bat to run docker container. Open browser and go to localhost:5001 (or 127.0.0.1:5001) and use.
