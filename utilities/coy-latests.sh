#!/usr/bin/env bash

base_path="/home/ubuntu/birks/ikt450_assignments/ikt450-4-convelutional_networks/output/"

scp -r "ubuntu@robins_server:$base_path/$1" "./latests/"

