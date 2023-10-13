from confluent_kafka import Consumer, KafkaException

bootstrap_servers = 'localhost:9092'
topic = 'quickstart-events'

conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'my_consumer_group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)

consumer.subscribe([topic])

try:
    while True:
        msg = consumer.poll(timeout=1000)  # Poll for messages, with a timeout in milliseconds

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event - not an error
                continue
            else:
                raise KafkaException(msg.error())

        print('Received message: {}'.format(msg.value().decode('utf-8')))

except KeyboardInterrupt:
    pass

finally:
    consumer.close()
