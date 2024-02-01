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

if __name__ == "__main__":
	file = open('logs1.txt','r')
	Lines = file.readlines()
	int_garb = 0
	float_garb = 0
	string_garb = "0"
	sr_no=1
	for line in Lines:
		obj_data = line.strip()
		sr_no+=1
		key = "key"
		str_obj_data = obj_data
		producer.produce(topic, key=key, value=str_obj_data, on_delivery=delivery_report)

producer.flush()

