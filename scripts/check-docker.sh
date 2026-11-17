#!/bin/sh
set -eu

skip_acceptance=0
case "$#:$*" in
    0:) ;;
    1:--tests-only) skip_acceptance=1 ;;
    *) printf 'Использование: sh scripts/check-docker.sh [--tests-only]\n' >&2; exit 2 ;;
esac

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
if ! docker info >/dev/null 2>&1; then
    colima start --cpu 2 --memory 4
fi
docker info >/dev/null

source_commit=$(git rev-parse HEAD)
image_tag=search-rl-env-checks:local
docker build --build-arg "SOURCE_COMMIT=$source_commit" -t "$image_tag" .

mkdir -p "$project_root/runs/docker"
artifact_directory=$(mktemp -d "$project_root/runs/docker/run-XXXXXXXX")
printf 'Результаты: %s\n' "$artifact_directory"
docker run --rm --network none --cpus "${SEARCH_RL_DOCKER_CPUS:-1}" --memory 3g --pids-limit 512 \
    --env "SEARCH_RL_SKIP_ACCEPTANCE=$skip_acceptance" \
    --mount "type=bind,source=$artifact_directory,target=/artifacts" "$image_tag"
