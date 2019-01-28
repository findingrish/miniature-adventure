{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    "tf-job-operator": {
      cloud: 'null',
      deploymentNamespace: 'null',
      deploymentScope: 'cluster',
      name: 'tf-job-operator',
      namespace: 'null',
      tfDefaultImage: 'null',
      tfJobImage: 'gcr.io/kubeflow-images-public/tf_operator:v0.3.0',
      tfJobUiServiceType: 'ClusterIP',
      tfJobVersion: 'v1alpha2',
    },
    "nfs-server": {
      name: 'nfs-server',
      namespace: 'null',
    },
    "nfs-volume": {
      capacity: '1Gi',
      mountpath: '/',
      name: 'nfsrishabh',
      namespace: 'null',
      nfs_server_ip: '10.27.243.46',
      storage_request: '1Gi',
    },
    "tf-mnistrishabhjob": {
      args: 'null',
      envs: 'TF_EXPORT_DIR=/mnt/export,TF_MODEL_DIR=/mnt/model',
      image: 'docker.io/rish691/aws-reco:v11',
      image_gpu: 'null',
      image_pull_secrets: 'null',
      name: 'tf-mnistrishabhjob',
      namespace: 'null',
      num_gpus: 0,
      num_masters: 1,
      num_ps: 1,
      num_workers: 1,
      volume_mount_path: 'null',
    },
    tfserving: {
      modelPath: '/mnt/export',
      modelStorageType: 'nfs',
      name: 'mnistrishabh',
      nfsPVC: 'nfsrishabh',
    },
  },
}