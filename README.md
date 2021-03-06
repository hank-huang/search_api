# search_api
dockerized smemantic search api using USEQA and bm25 ranker

## Overview
Created an API using Python, Flask, and Docker to handle search requests. Given a query and several candidate strings, which candidate string best matches the query? Supports two different requests, one which returns top 3 candidate strings based on an angular distance metric (favorable approximation of cosine distance), and the other which returns weighted scores of preprocessed candidate strings.

Credits to the folks at Tensorflow for the [USEQA algorithm](https://tfhub.dev/google/universal-sentence-encoder-qa/3), and dorian brown for the [bm25 ranker](https://github.com/dorianbrown/rank_bm25)! 

## Deployment Instructions

build docker image  

```
docker image build -t <image_name> <directory>
```

run docker image  

```
docker run -t -d -p 5000:5000 <image_name>:latest
```

query from api!  

Currently the api supports two GET requests that take in a json with two fields: query, responses  

query: what you want to search for  
responses: a list of candidate strings  

/search/USEQA --> top 3 candidate strings based on minimum angular distance will be returned  
/search/BM25 --> returns a list of weights using the bm25 algorithm  

Example:
```
curl --header "Content-Type: application/json" --request GET --data '{"query":"How old are you?","responses":["30","I am 30","hello_whale", "randomstring"]}' http:/localhost:5000/search/USEQA

```

## Challenges
### Big DockerFile
Tried to lessen it by using the alpine version of python — the problem is there is no tensorflow support for that version. 

### Model takes a long time to load
Main thing is that the model takes a long time to load because it has to download a ~500mb model using tensorflow_hub. Could’ve downloaded it manually but I figured it would’ve blown up the size of the API. I think it’s acceptable to eat the large cost of loading the model once, and requesting from the API without having to load the model in every single time.

### Had to learn about text embedding
Was challenging learning about text embedding algorithms and whatnot, especially after having spent some time away from NLP. Spent an hour or two refreshing myself on the concepts, and focused on learning the documentation.

### Time Crunch
Had to implement and design the ideas in essentially 7 or so hours with a lot going on — decided to focus on getting the project working, with ideas for improvement and limited testing.


## Further Improvements
### Speed up Model
Have the model live on the deployment server and run it from there

### Make DockerFile smaller
Lots of little optimizations could be made here, namely to limit the number of layers in the file. Would have to look more into that and learn more about docker in that sense!

### Automated Testing
Add in automated testing and unit tests with pytest. 

### Forms
Might be easier to enter data using forms instead of current json endpoints. 


## Naked Flask is not meant for Production — what alternative can we use to serve the API?
Gunicorn? From what I'm reading we could use it to serve our Flask application via a WSGI. Another alternative we could use is to serve it through Heroku.


## Compare embedding answers with the answer itself as context vs. empty context. Which approach produces better query results?
As far as I can tell they perform roughly the same. If there are relevant answers, they get pulled up — if no answers are relevant different ones do but then none of them really matter in that case. I'd have to investigate the mechanism behind the context — a process that I think would take a bit of time. 





