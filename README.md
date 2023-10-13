# Video-database-analytics

The commands to run kafka in background are : 

# Start the ZooKeeper service
$ bin/zookeeper-server-start.sh config/zookeeper.properties

# Start the Kafka broker service
$ bin/kafka-server-start.sh config/server.properties


## Commands to run on collab
!pip install norfair
!python3 data_ins.py input.mp4 --track-points bbox