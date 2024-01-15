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
	file = open('stream.txt','r')
	Lines = file.readlines()
	int_garb = 0
	float_garb = 0
	string_garb = "0"
	for line in Lines:
		obj_data = (line,int_garb,int_garb,string_garb,float_garb,float_garb,float_garb,float_garb)
		key = "key"
		str_obj_data = str(obj_data)
		producer.produce(topic, key=key, value=str_obj_data, on_delivery=delivery_report)

producer.flush()

