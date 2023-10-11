#!/usr/bin/env bash

# Remote server information
remote_user="ubuntu"
remote_host="robins_server"
remote_path="/home/ubuntu/birks/ikt450_assignments/ikt450-4-convelutional_networks/output/"

# Local destination directory
local_destination="./latests/"

echo "Before"
latest_folder=$(ssh "$remote_user@$remote_host" "find $remote_path -maxdepth 1 -type d | sort -n | tail -1 | cut -d ' ' -f2")
echo $latest_folder 
echo "$local_destination$(basename "$latest_folder")"

scp -r "$remote_user@$remote_host:$latest_folder" "$local_destination"

echo "After"


