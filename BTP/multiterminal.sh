#!/bin/bash

# Open a terminal and start a long-running process
gnome-terminal --tab -- /bin/bash -c "mvn clean package ; python3 ../database/kafka_ins.py ../database/input.mp4 --track-points bbox; exec bash"

# Open another terminal and monitor system resources
gnome-terminal --tab -- /bin/bash -c "sleep 20 ; mvn exec:java -Dexec.mainClass=btp.VideoAnalysis; exec bash"