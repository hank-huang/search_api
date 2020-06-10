import flask
from flask import request, jsonify
import tf
from nlp import preprocess
from rank_bm25 import BM25Okapi

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/search/USEQA', methods=['GET'])
def api_USEQA():
    query = request.json['query']
    corpus = request.json['responses']

    return jsonify(tf.get_predictions(query, corpus))


@app.route('/search/BM25', methods=['GET'])
def api_BM25():
    query = request.json['query']
    corpus = request.json['responses']

    tokenized_corpus = [preprocess(sentence) for sentence in corpus]
    tokenized_query = preprocess(query)

    bm25 = BM25Okapi(tokenized_corpus)
    weights = bm25.get_scores(tokenized_query)

    return jsonify(weights)


app.run()