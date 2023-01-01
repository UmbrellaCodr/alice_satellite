#!/bin/bash

cd /root

for i in *.wav; do python3 -m alice_satellite predict $i; done



