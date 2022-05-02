# Importing the packages for setting up kafka producer - consumer pipeline

from kafka import KafkaConsumer
from json import loads
import sys
import json
from kafka import KafkaProducer

# declaring bootstrap servers, topicName, producer

bootstrap_servers = ['localhost:9092']
topicName = 'json_kafka'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
#producer = KafkaProducer()

# sending a message from producer

ack = producer.send(topicName, json.dumps('bad').encode('utf-8'))
metadata = ack.get()
print(metadata.topic)
print(metadata.partition)

# let's check if the message is received by consumer.py
