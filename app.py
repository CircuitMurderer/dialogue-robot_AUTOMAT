from interaction import initialize, predict, Config
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

config = Config()
model, tokenizer = initialize(config)
history = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return predict(model, tokenizer,
                   config, history, user_text)


if __name__ == "__main__":
    app.run()
