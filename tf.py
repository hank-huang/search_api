import tensorflow as tf
import tensorflow_hub as hub
import simpleneighbors
import pickle

model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-qa/3')

print(tf.saved_model.contains_saved_model("./"))

print("model loaded")


def get_predictions(query, responses):
    encodings = model.signatures['response_encoder'](
      input=tf.constant(responses),
      context=tf.constant(responses))

    query_index = simpleneighbors.SimpleNeighbors(
      len(encodings['outputs'][0]), metric='angular')

    for idx, batch in enumerate(responses):
        query_index.add_one(batch, encodings['outputs'][idx])

    query_index.build()

    query_embedding = model.signatures['question_encoder'](tf.constant([query]))['outputs'][0]
    search_results = query_index.nearest(query_embedding, n=3)

    return search_results
