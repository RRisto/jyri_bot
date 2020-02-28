set APP_PATH=%cd%

docker stop jyri_bot_run
docker rm jyri_bot_run
docker run -it  ^
    -p 5001:5001 ^
    -v %APP_PATH%:/opt/fs ^
    --name jyri_bot_run ^
    jyri_bot