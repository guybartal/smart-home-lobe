import paho.mqtt.client as mqtt

client_name = "Lobe classifier"
default_topic = "classifier"
default_host = "localhost"
default_port = 1883

class Publisher:
    def __init__(self, host=default_host, port=default_port):
        self.client = mqtt.Client(client_name)
        self.client.connect(host, port)

    def publish(self, prediction, topic=default_topic):
        payload=prediction['Prediction']
        print("Publishing message, topic: "+topic+", payload: "+payload)
        self.client.publish(topic, payload)