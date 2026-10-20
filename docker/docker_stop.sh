#!/bin/bash

source ./docker_base.sh

function stop_docker() {
  echo "stop container in workspace: ${WS_DIR}"
  local docker_info=$(fetch_docker_info)
  local container_id=$(echo "${docker_info}" | awk -F'|' '{print $1}')
  if [ "${container_id}" != "" ]; then
    ${DOCKER_CMD} stop ${container_id}
  fi
}

stop_docker
