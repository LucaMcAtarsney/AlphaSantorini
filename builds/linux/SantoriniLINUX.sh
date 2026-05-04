#!/bin/sh
printf '\033c\033]0;%s\a' Santorini
base_path="$(dirname "$(realpath "$0")")"
"$base_path/SantoriniLINUX.x86_64" "$@"
