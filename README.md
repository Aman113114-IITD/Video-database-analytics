# Video-database-analytics

The commands to run kafka in background are : 

# Start the ZooKeeper service
$ bin/zookeeper-server-start.sh config/zookeeper.properties

# Start the Kafka broker service
$ bin/kafka-server-start.sh config/server.properties

# command to make a topic
bin/kafka-topics.sh --create --topic quickstart-events --bootstrap-server localhost:9092


## Commands to run on collab
!pip install norfair
!python3 data_ins.py input.mp4 --track-points bbox

# commands to run java flink cep and read data from kafka stream
mvn clean package
mvn exec:java -Dexec.mainClass=wikiedits.WikipediaAnalysis