#!/bin/bash

function _create_user_account() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"
  addgroup --gid "${gid}" "${group_name}"

  adduser --disabled-password --force-badname --gecos '' \
    "${user_name}" --uid "${uid}" --gid "${gid}"

  usermod -aG sudo "${user_name}"
  echo "${user_name}  ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

  local user_home="/home/${user_name}"
  cp /root/.bashrc /root/.vimrc /root/.gitconfig ${user_home}
  local user_torch_ck_dir="${user_home}/.cache/torch/hub/checkpoints"
  mkdir -p ${user_torch_ck_dir} && cp /opt/downloads/alexnet-owt-7be5be79.pth ${user_torch_ck_dir}/
  chown -R "${uid}:${gid}" ${user_home}/.*
  echo "source /opt/conda/etc/profile.d/conda.sh && conda activate main" >> ${user_home}/.bashrc
}

function setup_user_account_if_not_exist() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"
  if grep -q "^${user_name}:" /etc/passwd; then
    echo "User ${user_name} already exist. Skip setting user account."
    return
  fi  
  _create_user_account "$@"
}

function main() {
  local user_name="${DOCKER_USER}"
  local uid="${DOCKER_UID}"
  local group_name="${DOCKER_GROUP}"
  local gid="${DOCKER_GID}"

  if [ "${uid}" != "${gid}" ]; then
    echo "Warning: uid(${uid}) != gid(${gid}) found."
  fi
  if [ "${user_name}" != "${group_name}" ]; then
    echo "Warning: user_name(${user_name}) != group_name(${group_name}) found."
  fi

  setup_user_account_if_not_exist ${user_name} ${uid} ${group_name} ${gid}
}

main
