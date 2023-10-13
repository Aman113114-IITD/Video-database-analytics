from confluent_kafka import Producer

bootstrap_servers = 'localhost:9092'
topic = 'events'

conf = {
    'bootstrap.servers': bootstrap_servers,
}

def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed:', err)
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

producer = Producer(conf)

try:
    # Produce messages to the Kafka topic
    for i in range(10):
        key = 'key' + str(i)
        value = 'message' + str(i)
        producer.produce(topic, key=key, value=value, on_delivery=delivery_report)

    # Flush the producer to send messages to Kafka
    producer.flush()

except KeyboardInterrupt:
    pass

producer.flush()
