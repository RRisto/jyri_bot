## Jüri Ratas bot
Bot trained on Jüri Ratas (currently Estonian Prime Minister) interviews and talks in Riigikogu
 (also from times when Jüri was not Prime Minister). 
 
![Jüri Ratas](https://github.com/RRisto/jyri_bot/blob/master/static/ratas_small.png) 
 
Bot is using transformer architecture. 
Data used for training model is not comprehensive. 
Mostly collected manually and using API for Riigikogu.
Goal of this project is to have some fun and see if sequnece2sequence model could give 
human like answers. As Jüri is knwon for not giving direct answers some of the bot answers are not 
very different what you might when interviewing him.

### Train model
To train model use notebook 0.1_prepare_jyri_data.ipynb to prepare data and 
notebook 0.2_train_jyri.ipynb to train model.
Use GPU to seed up training. NB! in requirements.txt there is only CPU version of pytorch ()

### Use bot
To use bot have model files in models sub folder. Use build_dev.bat (you need to convert it.sh files
 if using linux) to build docker image.
Use run_dev.bat to run docker container. Open browser and go to localhost:5001 (or 127.0.0.1:5001) 
and use.

## Credits
Thanks [Samuel Lynn-Evans](https://towardsdatascience.com/@samuellynnevans) for 
transformer [tutorial](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec) and [code](https://github.com/SamLynnEvans/Transformer). 
I've slightly modified code and turned it to dockerised flask app.
