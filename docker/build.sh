#!/bin/bash

source ./docker_base.sh

my_http_proxy=""
my_https_proxy=""

while [ $# -gt 0 ]; do
  case "$1" in
    --http_proxy=*)
      my_http_proxy="${1#*=}"; shift ;;
    --https_proxy=*)
      my_https_proxy="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

echo "http_proxy: ${my_http_proxy}"
echo "https_proxy: ${my_https_proxy}"

function get_beijing_time() {
  local beijing_time
  local format="%Y%m%d%H%M%S"
  local tz="Asia/Shanghai"
  if command -v curl >/dev/null 2>&1; then
    local date_header
    date_header=$(curl -sS -m 3 -I https://www.baidu.com | grep -i '^Date:' | cut -d' ' -f2- || true)
    if [ -n "${date_header}" ]; then
      beijing_time=$(TZ="${tz}" date -d "${date_header}" +"${format}" 2>/dev/null || true)
      if [ -n "${beijing_time}" ]; then
        echo "${beijing_time}"
        return 0
      fi
    fi
  fi

  beijing_time=$(TZ="${tz}" date +"${format}")
  echo "${beijing_time}"
}

BEIJING_TIME=$(get_beijing_time)

cp ../requirements.txt ./copy/opt/requirements.main.txt
SCENE_TMP_IMG="tmp_i_scene"
${DOCKER_CMD} build \
  --build-arg MY_HTTP_PROXY=${my_http_proxy} \
  --build-arg MY_HTTPS_PROXY=${my_https_proxy} \
  -t ${SCENE_TMP_IMG} .
rm ./copy/opt/requirements.main.txt

SCENE_TMP_CONTAINER="tmp_c_scene_${BEIJING_TIME}"
${DOCKER_CMD} run \
    --name ${SCENE_TMP_CONTAINER} \
    --runtime=nvidia \
    --gpus all \
    ${SCENE_TMP_IMG} bash -c "/opt/install_cuda_relay.sh"

${DOCKER_CMD} commit ${SCENE_TMP_CONTAINER} ${DOCKER_REPO}:${BEIJING_TIME}
${DOCKER_CMD} rm ${SCENE_TMP_CONTAINER}

echo "build ${DOCKER_REPO}:${BEIJING_TIME} done"
