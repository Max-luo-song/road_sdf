#!/bin/bash

set -e

function path_to_name() {
  local path="$1"
  local name="${path//\//-}"
  name="${name#-}"
  echo "${name}"
}

DOCKER_REPO="scene/cuda12.1_py39_pyt231"
DOCKER_WS=/scene_reconstruction

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_NAME=$(path_to_name ${WS_DIR})

function docker_cmd() {
  local cmd="sudo docker"
  if docker ps >/dev/null 2>&1; then
    cmd="docker"
  fi
  echo ${cmd}
}

DOCKER_CMD=$(docker_cmd)

function fetch_docker_info() {
  echo $(${DOCKER_CMD} ps -a --filter "name=${DOCKER_NAME}" --format "{{.ID}}|{{.Status}}")
}

function get_newest_tag() {
  local tags=($(${DOCKER_CMD} images ${DOCKER_REPO} --format "{{.Tag}}" | sort -rn))
  local newest_tag="unknown"
  if [ ${#tags[@]} -gt 0 ]; then
    newest_tag="${tags[0]}"
  fi
  echo "${newest_tag}"
}

function get_image_name() {
  image_version="$1"
  echo "get_image_name recv version param: ${image_version}" >&2

  if [ "${image_version}" == "" ]; then
    image_version=$(get_newest_tag)
  fi

  local image_name="${DOCKER_REPO}:${image_version}"
  echo "will inspect docker image: ${image_name}" >&2
  if ! ${DOCKER_CMD} image inspect "${image_name}" > /dev/null 2>&1; then
    echo "image: ${image_name} not exists" >&2
    exit 1
  fi

  echo "${image_name}"
}
