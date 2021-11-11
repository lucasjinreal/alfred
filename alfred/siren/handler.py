import json
from functools import wraps
from logging import log
from loguru import Message
from paho import mqtt
from paho.mqtt import client as mqtt_client
import random
import platform
import uuid
from alfred.siren.topicgen import get_archives_messages_topic, get_archives_myid_topic, get_archives_rooms_topic, get_chatting_topic, get_events_topic, get_personal_events_topic, get_presence_topic
from alfred.utils.log import logger
from alfred.siren.models import ChatMessage, ContactChat, Invitation, InvitationMessage, InvitationMessageType, User, MessageType
import jsons

MQTT_URL = "manaai.cn"
PORT = 1883


class SirenClient:

    def __init__(self, user_acc, user_password) -> None:
        self.user_acc = user_acc
        self.user_password = user_password

        self.client_id = f'{platform.system()}_{uuid.uuid4()}'
        self.user = None
        self._connect()

    def _connect(self):
        self.client = mqtt_client.Client(self.client_id)
        self.client.username_pw_set(self.user_acc, self.user_password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(MQTT_URL, PORT)
    
    def loop(self):
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

    def join_my_events(self, myid):
        self.client.subscribe(get_personal_events_topic(myid))

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
            self.user = jsons.load(j, User)
            self.join_my_events(self.user.user_addr)
            logger.info(f'Welcome: {self.user.user_nick_name}')
        elif msg.topic.startswith('personalevents/'):
            # invitation auto agree
            invit = jsons.load(j, InvitationMessage)
            print(invit)
            logger.info(f'Received invitation: {invit.from_name}')
            self.response_to_invitation(invit.id, invit.from_id)
            self.on_received_invitation(invit)
        elif msg.topic.startswith('messages/'):
            m = jsons.load(j, ChatMessage)
            logger.info(f'Received ChatMessage: {m.type} {m.text} {m.from_name} {m.room_id}')
            self.publish_txt_msg('I got your message', m.room_id)
            self.on_received_chat_message(m)
        else:
            logger.info('unsupported msg.')

    def response_to_invitation(self, invi_id, sender_id):
        invit = InvitationMessage()
        invit.id = invi_id
        invit.from_id = sender_id
        invit.type = MessageType.EventInvitationResponseAccept
        invit.invitation_type = InvitationMessageType.REQUEST_RESPONSE
        j = jsons.dump(invit)
        self.client.publish(get_personal_events_topic(sender_id), j)

    def on_received_invitation(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # we receive msg from mqtt conn, then convert it into ChatMessage
            return func(*args, **kwargs)
        return wrapper

    def on_received_chat_message(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # we receive msg from mqtt conn, then convert it into ChatMessage
            return func(*args, **kwargs)
        return wrapper

    def publish_txt_msg(self, txt, room_id):
        msg = ChatMessage()
        msg.room_id = room_id
        msg.text = txt
        msg.type = MessageType.ChatText
        j = jsons.dump(msg)
        t = get_chatting_topic(room_id)
        self.client.publish(t, j)
