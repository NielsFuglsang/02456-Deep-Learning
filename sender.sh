#!/bin/bash
sed "s+name+$1+g" < job.sh | bsub
