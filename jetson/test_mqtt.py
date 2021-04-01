from publisher import Publisher

publisher = Publisher()
prediction = {'Prediction': 'Hello MQTT'}
publisher.publish(prediction)