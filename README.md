# RecoStream — Real-Time ML Recommendation System

A real-time movie recommender built on **MovieLens-25M** (25 million ratings, 162,541 users, 59,047 movies).
- **Candidates:** collaborative filtering generates them.
- **Ranking:** a neural ranker re-orders them using the user's interaction history and behavioral features.
- **Streaming:** interaction events flow through Kafka, and a consumer group updates each user's features in Redis as they arrive.
- **Serving:** a FastAPI service returns recommendations with micro-batched inference.
- **Model management:** models are versioned in S3 and only promoted after automated evaluation.
- **Monitoring:** Prometheus tracks both latency and live recommendation quality.

**Stack:** Python · PyTorch · scikit-learn / SciPy · Kafka (confluent-kafka) · Redis · FastAPI · AWS S3 / EC2 (boto3, Terraform) · Docker · Prometheus

## Architecture

```
 offline (python -m recostream.train)                       online
 ───────────────────────────────────                        ──────────────────────────────────────────────
 MovieLens-25M → temporal split                              clients ──► FastAPI /recommend/{user}
   ├─ BPR matrix factorization (PyTorch, GPU)                     │ micro-batcher (≤64 users / 3 ms)
   ├─ neural ranker on MF candidates + features                   │  1. Redis: online features (1 round trip)
   ├─ evaluate NDCG@10 vs popularity baselines                    │  2. MF scores vs all 59k items, top-200
   └─ S3 registry: models/{version}/…, eval gate                  │  3. ranker on 200 candidates → top-k
            │ production.json                                     ▼
            └──────────── model download / hot reload ──►  Redis feature store ◄── Kafka consumer group
                                                                                    (N processes)
                                                               interaction events ──► Kafka topic
```

**Collaborative filtering** (`models/mf.py`)
- BPR matrix factorization with 64-dimensional embeddings, trained on 23.4M implicit interactions.
- Negatives are half uniform and half popularity-sampled, which stops the model from simply learning popularity.
- Candidate generation scores all 59,047 movies, masks the ones already seen, and keeps the top 200.

**Neural ranker** (`models/ranker.py`, `features.py`)
- An MLP over the user embedding, the item embedding, their element-wise product, and 12 behavioral features:
  - MF score and rank
  - Item popularity (all-time and recent), item mean rating, release year
  - User activity and mean rating
  - User–genre affinity
  - Similarity of the candidate to the user's 10 most recent items (mean, max, and to the last item)
- Trained listwise (softmax cross-entropy over each user's candidate list).
- `pair_features` is the same function offline and online, so training and serving can't drift apart.

**Evaluation protocol** (`data.py`, `train.py`)
- Each user's interactions are split by time into history, validation (next 5), and test (last 5).
- MF trains on history. The ranker learns to predict validation items from history-based features.
- The test evaluation uses features from history + validation, mirroring production, where streamed features are fresher than the last model training.

**Streaming** (`stream/`)
- The producer replays interactions into Kafka, keyed by user so each user's events stay in order.
- The consumer runs as a group of processes. Each processes batches with two pipelined Redis round trips and commits offsets afterwards (at-least-once).
- Per batch it updates the user's recent-item list, rating statistics, genre counts, and a trending-items counter.
- It also checks whether each event's movie was in that user's last served top-10, giving **online recommendation quality in real time**.

**Serving** (`serving/`)
- Concurrent requests are grouped into micro-batches (batch prediction): one Redis round trip, one candidate matrix multiply, and one ranker forward pass per batch.
- Serving runs on CPU with 4 uvicorn workers.
- A user with no MF embedding gets one built from their recent items; a user with no history gets popularity.

**Model registry and gate** (`registry.py`)
- Every training run uploads a versioned artifact (embeddings, item features, ranker weights, metrics) to S3.
- A version becomes production only if its NDCG@10 doesn't regress against the current production model.
- The API loads the production model at startup and can hot-reload it with `POST /admin/reload`.

## Results

**Setup:** an Apple M4 laptop (10 cores) with everything running locally: Kafka, Redis, the API, consumers, and the load generator.

### Offline ranking quality (MovieLens-25M, 20,000 random test users)

| Model | NDCG@10 | Recall@10 | HitRate@10 |
|---|---:|---:|---:|
| Popularity, all-time | 0.0389 | 0.0459 | 0.172 |
| Popularity, last 30 days before each user's cutoff | 0.0599 | 0.0712 | 0.249 |
| Matrix factorization (BPR) alone | 0.0388 | 0.0486 | 0.185 |
| **MF candidates + neural ranker** | **0.0902** | **0.1049** | **0.362** |

- **The ranker beats both baselines:** NDCG@10 is **+132%** over all-time popularity and **+51%** over the stronger time-aware popularity baseline.
- **Where the gain comes from:** mostly the history-aware features. MF alone does no better than popularity in this temporal setting.
- **Reproducible:** two training runs gave NDCG@10 of 0.0900 and 0.0902.
- Training plus evaluation takes about 4.5 minutes on the M4 GPU. [Details](results/offline.md).

### Streaming feature updates (Kafka → Redis)

Throughput when draining 200,000 pre-loaded events ([full table](results/stream.md)):

| Partitions | Consumer processes | Events/sec |
|---:|---:|---:|
| 1 | 1 | 27,565 – 32,001 |
| 4 | 2 | 29,923 – 30,833 |
| 4 | 4 | 44,345 – 45,996 |
| 8 | 8 | **76,323 – 80,088** |

Each range spans batch sizes of 100 and 1,000. The producer writes about 400,000 events/s.

### Serving latency

Open-loop load generated with `oha` at fixed request rates, each request for a random user ([raw](results/serving.md)):

| Request rate | p50 | p95 | p99 |
|---:|---:|---:|---:|
| 250 req/s | 6.1 ms | 10.3 ms | 11.9 ms |
| 500 req/s | 7.5 ms | 11.8 ms | 14.1 ms |
| 1,000 req/s | 9.2 ms | 13.9 ms | 16.0 ms |
| 2,000 req/s | 11.2 ms | 17.5 ms | 20.6 ms |
| 3,000 req/s | 24.1 ms | 144.5 ms | 258.9 ms |

**Saturation:** this laptop handles about 2,000–3,000 req/s before queueing dominates. One earlier run held p95 at 33 ms at 3,000 req/s.

**Where the time goes:** one micro-batch takes about 1.9 ms of compute (0.35 ms Redis features, 0.9 ms candidates, 0.6 ms ranking).

### End-to-end online quality

The pipeline was run end to end:
1. Top-10 lists were served through the API to 5,000 random test users.
2. Those users' real next 25,000 interactions were streamed through Kafka.
3. The consumer scored each event in real time against what had been served.

Result: online **HitRate@10 = 0.346**, compared with **0.362 offline** on the same metric ([details](results/online.md)). This confirms the serving path reproduces offline quality, and the same signal is exported continuously as a Prometheus metric.

## Running locally

```bash
brew install kafka redis oha
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
# MovieLens-25M: https://grouplens.org/datasets/movielens/25m/ → ../datasets/ml-25m (or set RECO_RAW_DIR)

scripts/infra.sh start                              # Kafka (KRaft), Redis, local S3 endpoint on :9000
export S3_ENDPOINT=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
.venv/bin/python -m recostream.train                # train, evaluate, upload to S3, gate + promote
.venv/bin/python -m recostream.bootstrap            # load online features into Redis

PROMETHEUS_MULTIPROC_DIR=$(mktemp -d) .venv/bin/uvicorn recostream.serving.app:app --workers 4 --port 8000 &
.venv/bin/python -m recostream.stream.consumer --workers 4 &
.venv/bin/python -m recostream.stream.producer --split test --rate 5000 &

curl localhost:8000/recommend/42?k=10
curl localhost:8000/online-quality

.venv/bin/python -m pytest tests/                   # unit + Redis + registry-gate tests
.venv/bin/python scripts/loadtest.py                # serving latency sweep
.venv/bin/python scripts/bench_stream.py --model-npz artifacts/<version>/model.npz
.venv/bin/python scripts/e2e_online.py              # online vs offline quality
```

## Running with Docker

The whole stack runs in containers: Kafka, Redis, the API, the consumer group, Prometheus, and (locally) an S3-compatible endpoint.

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml --profile local-s3 up -d --build kafka redis s3 prometheus
# publish a trained model to the containerized S3 (exposed on :9000), then:
docker compose -f docker-compose.yml -f docker-compose.local.yml --profile local-s3 run --rm bootstrap
docker compose -f docker-compose.yml -f docker-compose.local.yml --profile local-s3 up -d api consumer
docker compose -f docker-compose.yml -f docker-compose.local.yml --profile local-s3 --profile jobs run --rm producer
```

This was verified end to end. The API served the production model pulled from S3. The containerized producer streamed 50,000 events through Kafka, and the consumer group processed all of them. Prometheus scraped both the API and the consumer.

## Deploying to AWS (EC2 + S3)

`deploy/terraform` creates:
- a versioned, private **S3 bucket** for model artifacts
- an **IAM role** that lets the instance read and write that bucket, so no keys are stored on the box
- a security group open only to your IP
- an **EC2 instance** that installs Docker on first boot

```bash
cd deploy/terraform && terraform init
terraform apply -var key_name=<ec2-keypair> -var allowed_cidr=<your-ip>/32
S3_BUCKET=$(terraform output -raw bucket) python -m recostream.train   # publish a model to S3
../../scripts/deploy_ec2.sh $(terraform output -raw public_ip) ~/.ssh/<key>.pem
```

On the instance, `docker compose` runs Kafka, Redis, the API, the consumer group, and Prometheus. Models come from S3 through the instance role. The Docker stack is verified locally (see above). The Terraform hasn't yet been applied to a real AWS account. Every benchmark number above comes from the local, non-container setup.

### Prometheus metrics

| Metric | Meaning |
|---|---|
| `reco_request_seconds` | End-to-end `/recommend` latency (p50/p95/p99) |
| `reco_stage_seconds{stage}` | Per-batch feature fetch, candidate generation, and ranking time |
| `reco_batch_size` | Users per inference micro-batch |
| `reco_model_info{version}` | Active model version (value is its offline NDCG@10) |
| `reco_stream_events_total{worker}` | Events processed per consumer process |
| `reco_online_hits_total`, `reco_online_events_with_recs_total` | Live recommendation quality (online hit rate) |

## Limitations

- The local S3 endpoint is a mock (moto). In production the same boto3 code talks to real S3.
- User MF embeddings refresh only when the model is retrained. Between retrains, real-time behavior reaches the ranker through the streamed features (recent items, genre profile, activity).
- Consumer delivery is at-least-once: a replayed event can be counted twice in the statistics.
- Everything was benchmarked on one laptop, so the load generator, API, Kafka, and Redis all share 10 cores.
