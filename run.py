# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

from transformer.Trainer import Trainer

app = Flask(__name__)

trainer = Trainer.load_model('models/jyri_bot_v1')


@app.route('/', methods=['GET'])
def kysi():
    return render_template('bot.html')


@app.route('/get_jyri_answer', methods=['GET'])
def get_jyri_answer():
    text = request.args.get("text")
    try:
        bot_response = trainer.predict(str(text))
    except Exception as e:
        print(f'Exception {e}')
        bot_response = 'Aitäh!'
    return bot_response


@app.route('/about')
def hello_world():
    return 'Hello World! Proov'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
