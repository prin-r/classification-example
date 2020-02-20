from flask import Flask, jsonify, request, abort

# from train.predict import process_text
from utils import process_text
from train import Sentiment

app = Flask(__name__)
sentiment = Sentiment()


@app.route("/api/analyze")
def analyze():
    text = request.args.get("text")
    if text is None or text.strip() == "":
        return jsonify(success=False)

    try:
        result = sentiment.predict(text)
        return jsonify(success=True, sentiment=result[0])
    except:
        return jsonify(success=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000", debug=True)
