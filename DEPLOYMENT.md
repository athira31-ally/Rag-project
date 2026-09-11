# Deployment walkthrough (EKS + Kong)

A generic build → push → deploy flow for running this app the way the
production system it's modeled on runs: containerized, on EKS, behind a
Kong gateway. Every name below (`<CLUSTER_NAME>`, `<ACCOUNT_ID>`, etc.) is a
placeholder — swap in your own AWS account/cluster details.

## 1. Build and push the image

```bash
docker build -t talkument-rag-api:latest -f docker/Dockerfile .

aws ecr get-login-password --region <REGION> \
  | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

docker tag talkument-rag-api:latest \
  <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/talkument-rag-api:latest

docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/talkument-rag-api:latest
```

(Swap `docker` for `podman` throughout if that's your build tool of choice —
the commands are otherwise identical.)

## 2. Point kubectl at your cluster

```bash
aws eks update-kubeconfig --region <REGION> --name <CLUSTER_NAME>
kubectl get nodes
kubectl create namespace talkument-rag
```

## 3. Configuration and secrets

Create a ConfigMap for the non-secret settings in `.env.example` and a
Secret for the rest (`OPENAI_API_KEY`, `GOOGLE_CLIENT_SECRET`,
`JWT_SECRET`, DB/Redis credentials if you're not using the defaults):

```bash
kubectl create configmap talkument-rag-config \
  --from-env-file=.env -n talkument-rag

kubectl create secret generic talkument-rag-secrets \
  --from-literal=OPENAI_API_KEY=<value> \
  --from-literal=JWT_SECRET=<value> \
  -n talkument-rag
```

Never commit a real `.env` or secret manifest — `.gitignore` already
excludes `.env`.

## 4. Deploy

```bash
kubectl apply -f k8s/deployment.yaml -n talkument-rag
kubectl get pods -n talkument-rag
kubectl get svc -n talkument-rag
```

`k8s/deployment.yaml` includes a `Deployment`, a `Service`, and an HPA that
scales 2–10 replicas on CPU utilization.

## 5. Put it behind Kong

```bash
helm repo add kong https://charts.konghq.com
helm repo update
helm install kong kong/kong -n kong --create-namespace

kubectl apply -f k8s/kong-route-example.yaml -n talkument-rag
```

`k8s/kong-route-example.yaml` shows the pattern used in production: a
path-based route (`/api/rag`) plus a rate-limiting plugin. Extend it with
whatever additional plugins (auth, logging, CORS) your deployment needs.

## 6. Roll out an update

```bash
docker build -t talkument-rag-api:latest -f docker/Dockerfile .
docker tag talkument-rag-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/talkument-rag-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/talkument-rag-api:latest

kubectl rollout restart deployment/talkument-rag-api -n talkument-rag
kubectl rollout status deployment/talkument-rag-api -n talkument-rag
```

## Troubleshooting

```bash
kubectl describe pod <pod-name> -n talkument-rag   # check Events for scheduling/pull errors
kubectl logs -n talkument-rag deployment/talkument-rag-api --tail=100
kubectl top pods -n talkument-rag                   # resource usage vs. requests/limits
```
