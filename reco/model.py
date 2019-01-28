import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import csv
import pprint
import os

EXPORT_DIR = os.getenv("TF_EXPORT_DIR", "/Users/rishabh.singh/bootcamp_3_12_2018/aws-recommender/reco")
TF_MODEL_DIR = os.getenv("TF_MODEL_DIR", "/Users/rishabh.singh/bootcamp_3_12_2018/aws-recommender")
MACHINE_TAG_PATH = '/opt/awsMachineTag.csv'
DATA_PATH = '/opt/awsMachineData.csv'
#MACHINE_TAG_PATH = "./awsMachineTag.csv"
#DATA_PATH = "./awsMachineData.csv"

print("Starting 1")

X_FEATURE = 'x' 
def read():
    points = genfromtxt(DATA_PATH, delimiter=',')
    col_mean = points.mean(axis=0)
    col_sdev =  points.std(axis=0)
    points = (points-col_mean)
    points = points/col_sdev 
    points = np.delete(points, [2,6,8,9,10,11,14,18,22,26,30,34,38,42,46], axis=1)
    col_mean = np.nanmean(points, axis=0)
    inds = np.where(np.isnan(points))
    points[inds] = np.take(col_mean, inds[1])
    return points


def serving_input_receiver_fn():
    inputs = {X_FEATURE: tf.placeholder(tf.float32, [1, 33])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main(_):
    points = read()
    points = np.float32(points)
    
    print("Starting 2")
    num_clusters = 6
    sess = tf.Session()
    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, model_dir=TF_MODEL_DIR, use_mini_batch=False)
    
    print("Starting 3")
    
    input_fn = tf.estimator.inputs.numpy_input_fn({X_FEATURE: points}, shuffle=True)
    train_spec = tf.estimator.TrainSpec(
        input_fn, max_steps=15)
    
    export_final = tf.estimator.FinalExporter(
        EXPORT_DIR, serving_input_receiver_fn=serving_input_receiver_fn)
    
    eval_spec = tf.estimator.EvalSpec(input_fn, exporters=export_final,
                                      steps=1)
  
    tf.estimator.train_and_evaluate(kmeans, train_spec, eval_spec)
    print("Starting 3")

    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    i = 1
    tags = {}
    with open(MACHINE_TAG_PATH) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            tags[i] = row[0]
            i = i + 1
    
    cluster_content = {}
    cluster_count = {}
    
    for i, point in enumerate(points):
        cluster_index = cluster_indices[i]
        machineType = tags[i+1]
        if cluster_index not in cluster_count:
            cluster_count[cluster_index] = 1
            cluster_content[cluster_index] = {}
        else:
            cluster_count[cluster_index] = cluster_count[cluster_index] + 1
            
        if machineType in cluster_content[cluster_index]:
            cluster_content[cluster_index][machineType] = cluster_content[cluster_index][machineType] + 1
        else:
            cluster_content[cluster_index][machineType] = 1
    
    #write to file
    done = []
    print(cluster_content)
    path = str(EXPORT_DIR + '/clusterContent.txt')
    with open(path,  mode='w') as f:
        for i in cluster_indices:
            cluster_index = cluster_indices[i]
            if cluster_index not in done:
                f.write("%s\n" % cluster_content[cluster_index])
                done.append(cluster_index)
    

    sess.close()
    
if __name__ == '__main__':
    tf.app.run()