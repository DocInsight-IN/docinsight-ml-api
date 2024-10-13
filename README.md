# DocInsight Machine Learning Backend APIs

This repo contains the machine learning microservice for the DocInsight
Application Stack

## Setup

- Create a k8s cluster on GKE. We use 8 nodes (each with 2vCPU and 4GBs RAM) for
  the cluster.
- Create a service account key (JSON). Grant only the required roles for the
  project to the service account. For Example, for this service we grant it
  permissions: Storage Admin, GKE Developer, GCR Developer.
- Create a secret named `GCP_CRENDETIALS` on the Github Secrets
  (Repo->Settings->Secrets) and paste the service account key file into the
  secret
- Configure the storage bucket related permissions for the service account

```shell
$ export PROJECT_ID=<PROJECT_ID>
$ export ACCOUNT=<ACCOUNT>

$ gcloud -q projects add-iam-policy-binding ${PROJECT_ID} \
    --member=serviceAccount:${ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/storage.admin

$ gcloud -q projects add-iam-policy-binding ${PROJECT_ID} \
    --member=serviceAccount:${ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/storage.objectAdmin
    
$ gcloud -q projects add-iam-policy-binding ${PROJECT_ID} \
    --member=serviceAccount:${ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/storage.objectCreator
```

- If you are on the `prod` branch already then upon a new push, the workflow
  defined in `.github/workflows/deployment_prod.yaml` should automatically run.

## Notes

- Since we use CPU-based pods withing the k8s cluster, we use ONXX optimizations
  for speedups. In future, we plan to include TensorRT optimization for
  GPU-based pods
- We use Kustomize to manage the deployment on k8s
- We conduct load-testing for varying number of workers, RAM, nodes, etc. (Load
  testing is under the `locust` directory)
