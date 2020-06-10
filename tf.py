import tensorflow as tf
import tensorflow_hub as hub
import simpleneighbors

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-qa/3'
model = hub.load(model_url)

print("model loaded")


def get_predictions(query, responses):
    encodings = model.signatures['response_encoder'](
      input=tf.constant(responses),
      context=tf.constant(responses))

    index = simpleneighbors.SimpleNeighbors(
      len(encodings['outputs'][0]), metric='angular')

    for batch_index, batch in enumerate(responses):
        index.add_one(batch, encodings['outputs'][batch_index])

    index.build()

    query_embedding = model.signatures['question_encoder'](tf.constant([query]))['outputs'][0]
    search_results = index.nearest(query_embedding, n=3)

    return search_results
