
### End-to-end online evaluation

Served top-10 to 5,000 random test users through the API, then streamed their real next 25,000 interactions through Kafka; the consumer scored hits in real time.

| Metric | Value |
|---|---:|
| Events streamed / processed | 25,000 / 25,000 |
| Events whose item was in the served top-10 | 2,525 (10.10%) |
| Online HitRate@10 (user had ≥1 of next 5 items in served top-10) | 0.3462 |
| Wall time (serve + stream + process) | 8.5 s |
