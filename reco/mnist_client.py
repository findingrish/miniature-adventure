#!/usr/bin/env python2.7
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations


TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))

channel = implementations.insecure_channel(
    TF_MODEL_SERVER_HOST, TF_MODEL_SERVER_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "mnistrishabh"
request.model_spec.signature_name = "serving_default"
w = np.float32(np.random.rand(1,33))
request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(w, shape=[1,33]))

result = stub.Predict(request, 10.0)  # 10 secs timeout

print(result)