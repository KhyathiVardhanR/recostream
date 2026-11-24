#!/usr/bin/env bash
# Local infrastructure: Kafka (KRaft, single broker), Redis, and an S3-compatible endpoint.
#   scripts/infra.sh start | stop | topic NAME PARTITIONS | wipe
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${RECO_INFRA_DIR:-$ROOT/data/infra}"
KAFKA_HOME="$(brew --prefix kafka)"
mkdir -p "$RUN/logs"

kafka_cfg() {
  cat >"$RUN/kraft.properties" <<EOF
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@127.0.0.1:9093
listeners=PLAINTEXT://127.0.0.1:9092,CONTROLLER://127.0.0.1:9093
advertised.listeners=PLAINTEXT://127.0.0.1:9092
controller.listener.names=CONTROLLER
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
log.dirs=$RUN/kafka
num.partitions=8
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
EOF
}

case "${1:-}" in
start)
  if [ ! -d "$RUN/kafka" ]; then
    kafka_cfg
    "$KAFKA_HOME/bin/kafka-storage" format -t "$("$KAFKA_HOME/bin/kafka-storage" random-uuid)" \
      -c "$RUN/kraft.properties" --standalone >/dev/null
  fi
  KAFKA_HEAP_OPTS="-Xmx2G -Xms1G" nohup "$KAFKA_HOME/bin/kafka-server-start" "$RUN/kraft.properties" \
    >"$RUN/logs/kafka.log" 2>&1 &
  echo $! >"$RUN/kafka.pid"
  redis-server --port 6379 --save "" --appendonly no --daemonize yes --pidfile "$RUN/redis.pid" \
    --logfile "$RUN/logs/redis.log"
  # S3-compatible endpoint for the model registry (moto's S3 server; on AWS
  # the registry talks to real S3 and S3_ENDPOINT is left unset).
  nohup "$ROOT/.venv/bin/moto_server" -H 127.0.0.1 -p 9000 >"$RUN/logs/s3.log" 2>&1 &
  echo $! >"$RUN/s3.pid"
  for _ in $(seq 1 60); do
    "$KAFKA_HOME/bin/kafka-topics" --bootstrap-server 127.0.0.1:9092 --list >/dev/null 2>&1 && break
    sleep 1
  done
  echo "infra up: kafka 127.0.0.1:9092, redis 127.0.0.1:6379, s3 http://127.0.0.1:9000"
  ;;
topic)
  "$KAFKA_HOME/bin/kafka-topics" --bootstrap-server 127.0.0.1:9092 --delete --topic "$2" >/dev/null 2>&1 || true
  sleep 1
  "$KAFKA_HOME/bin/kafka-topics" --bootstrap-server 127.0.0.1:9092 --create --topic "$2" --partitions "$3" \
    --replication-factor 1 >/dev/null
  echo "topic $2 with $3 partitions"
  ;;
stop)
  for f in "$RUN"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; done
  echo stopped
  ;;
wipe)
  "$0" stop >/dev/null 2>&1 || true
  rm -rf "$RUN"
  ;;
*)
  echo "usage: $0 {start|stop|topic NAME PARTITIONS|wipe}" >&2
  exit 2
  ;;
esac
