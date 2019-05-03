#!/bin/sh

echo "Formatting C files"
clang-format -i src/*.c
echo "Formatting Header files"
clang-format -i include/*.h
