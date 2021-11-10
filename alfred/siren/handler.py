import json
from functools import wraps
from logging import log
from paho import mqtt
from paho.mqtt import client as mqtt_client
import random
import platform
import uuid
from alfred.siren.topicgen import get_archives_messages_topic, get_archives_myid_topic, get_archives_rooms_topic, get_chatting_topic, get_events_topic, get_presence_topic
from alfred.utils.log import logger
from alfred.siren.models import ContactChat, User

MQTT_URL = "manaai.cn"
PORT = 1883


class SirenClient:

    def __init__(self, user_acc, user_password) -> None:
        self.user_acc = user_acc
        self.user_password = user_password

        self.client_id = f'{platform.system()}_{uuid.uuid4()}'
        self._connect()

    def _connect(self):
        self.client = mqtt_client.Client(self.client_id)
        self.client.username_pw_set(self.user_acc, self.user_password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(MQTT_URL, PORT)
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        print('connected.. status: ', rc)
        self.subscribe_topics(client)

    def on_disconnected(self, client, userdata, flags, rc=0):
        print('disconnected...')

    def subscribe_topics(self, client: mqtt_client):
        client.subscribe(get_archives_rooms_topic(self.client_id))
        client.subscribe(get_archives_messages_topic(self.client_id))
        client.subscribe(get_archives_myid_topic(self.client_id))

    def join_room(self, room_id):
        self.client.subscribe(get_chatting_topic(room_id))
        self.client.subscribe(get_events_topic(room_id))

    def join_contact_presence(self, contact_id):
        self.client.subscribe(get_presence_topic(contact_id))

    def on_message(self, client, userdata, msg):
        j = json.loads(msg.payload.decode('utf-8'))
        logger.info(
            '[Msg arrived] topic: {}, payload: {}'.format(msg.topic, j))
        if msg.topic.startswith('archivesrooms/'):
            # join room
            if j != None:
                contacts = [ContactChat(**i) for i in j]
                for c in contacts:
                    self.join_room(c.room_id)
                    self.join_contact_presence(c.id)
        elif msg.topic.startswith("archivesmyid/"):
            # get my id
            u = User()
            u.from_json(j)
            logger.info(f'Welcome: {u.user_nick_name}')
        elif msg.topic.startwith('personalevents'):
            # invitation auto solove
            pass

    def on_msg_arrived(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # we receive msg from mqtt conn, then convert it into ChatMessage

            return func(*args, **kwargs)
        return wrapper

    def publish_msg(self, msg):
        pass
