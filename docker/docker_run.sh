#!/bin/bash

set -e

source ./docker_base.sh

IMAGE_VERSION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --imgv=*)
      IMAGE_VERSION="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

export DOCKER_USER="$(id -u -n)"
export DOCKER_UID="$(id -u)"
export DOCKER_GROUP="$(id -g -n)"
export DOCKER_GID="$(id -g)"

DEV_INSIDE="in-scene-docker"

function run_docker() {
  local local_host="$(hostname)"
  local display="${DISPLAY:-:0}"
  local image_name=$(get_image_name "${IMAGE_VERSION}")
  local container_id=$(${DOCKER_CMD} run -idt \
      --name ${DOCKER_NAME} \
      --runtime=nvidia \
      --gpus all \
      --hostname ${DEV_INSIDE} \
      --add-host "${DEV_INSIDE}:127.0.0.1" \
      --add-host "${local_host}:127.0.0.1" \
      --pid=host \
      -e DISPLAY="${display}" \
      -e DOCKER_USER=${DOCKER_USER} \
      -e DOCKER_UID=${DOCKER_UID} \
      -e DOCKER_GROUP=${DOCKER_GROUP} \
      -e DOCKER_GID=${DOCKER_GID} \
      -v ${WS_DIR}:${DOCKER_WS} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /etc/localtime:/etc/localtime:ro \
      -v /data/sfs_turbo:/Data \
      -w ${DOCKER_WS} \
      ${image_name} /bin/bash)

  echo ${container_id}
}

function main() {
  local docker_info=$(fetch_docker_info)
  local container_id=$(echo "${docker_info}" | awk -F'|' '{print $1}')
  local container_status=$(echo "${docker_info}" | awk -F'|' '{print $2}')

  local start_new_container=0

  case "${container_status}" in
    Up*) echo "container is running..." ;;
    *)
      if [ "${container_id}" != "" ]; then
        ${DOCKER_CMD} rm -f ${container_id}
      fi
      container_id=$(run_docker)
      if [ "${container_id}" == "" ]; then
        echo "start docker error in workspace: ${WS_DIR}"
        exit 1
      fi
      ${DOCKER_CMD} exec -u root "${container_id}" bash -c '/opt/docker_start_user.sh && /opt/set_torch_cuda_arch.sh'
    ;;
  esac

  ${DOCKER_CMD} exec -u "${DOCKER_USER}" -it ${container_id} /bin/bash -l
}

main
