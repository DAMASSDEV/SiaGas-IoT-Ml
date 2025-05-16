import pandas as pd
import paho.mqtt.client as mqtt
import time

class DataDummy:
    def __init__(self):
        self.lpg = 0
        self.butana = 0
        self.metana = 0
        self.sensor_values = []
        self.MAX_STACK_SIZE = 10

    def handle_message(self, payload):
        print("Data Server Diterima: ", payload)


        try:
            parts = payload.split(".")
            if len(parts) == 3:
                self.lpg = f"{int(parts[0]):02}"
                self.butana = f"{int(parts[1]):02}"
                self.metana = f"{int(parts[2]):02}"

                sensor_value = f"{self.lpg}.{self.butana}.{self.metana}"

                sensor_value.append(self.sensor_values)


                if len(self.sensor_values) > self.MAX_STACK_SIZE:
                    self.sensor_values.pop(0)

                df = pd.DataFrame(self.sensor_values, columns =  ["sensor_value"])
                df.to_csv("testing_server.csv", index=False)

                print(f"Data added: {self.sensor_values}")
            else:
                print("Invalid format: must be '00.00.00'")
        except Exception as e:
            print("Error parsing data:", e)

data_dummy = DataDummy()


def on_connect(client, userdata, flags, rc):
    print("Server Connected:", rc)
    client.subscribe("sensor/value")

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    data_dummy.handle_message(payload)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("mqtt.yusufghazali.com", 1883, 60)
client.username_pw_set("danar", "13112004")

client.loop_start()

try:
    print("📡 Receiving data....")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
        print("⏹️ Data collection stopped.")
        client.loop_stop()
        client.disconnect()
