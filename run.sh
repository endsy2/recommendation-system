#!/usr/bin/env bash
set -e

CMD="${1:-help}"

case "$CMD" in
  up)
    echo "Starting Song Search stack in Docker..."
    docker compose up -d --build
    echo ""
    echo "All services started!"
    echo "- Web App:        http://localhost:8000"
    echo "- Milvus Health:  http://localhost:9091/healthz"
    echo "- MinIO Console:  http://localhost:9001 (minioadmin / minioadmin)"
    ;;
  migrate)
    echo "Migrating FAISS sample into Milvus..."
    docker compose run --rm migrate
    ;;
  build|build-sample)
    echo "Building 1,000-song sample index into Milvus..."
    docker compose run --rm build-sample
    ;;
  test)
    echo "Running tests in Docker..."
    docker compose run --rm test
    ;;
  stop)
    echo "Stopping containers (preserving data)..."
    docker compose stop
    ;;
  down)
    echo "Stopping and removing containers (preserving data volumes)..."
    docker compose down
    ;;
  logs)
    docker compose logs -f web
    ;;
  ps)
    docker compose ps
    ;;
  restart)
    docker compose restart
    ;;
  help|*)
    echo "Song Search - Docker Helper"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  up           Build and start all services (Milvus + MinIO + etcd + Web App)"
    echo "  migrate      Migrate artifacts/sample into Milvus (songs_sample)"
    echo "  build        Build 1,000-song sample index from CSV into Milvus"
    echo "  test         Run test suite inside the Docker container"
    echo "  logs         Follow FastAPI web application logs"
    echo "  ps           Check status of all containers"
    echo "  stop         Stop running containers (keeps your data)"
    echo "  restart      Restart running containers"
    echo "  down         Take down containers (keeps your data volumes)"
    ;;
esac
