
from functools import wraps
from alfred.siren.handler import SirenClient
from alfred.siren.models import ChatMessage, InvitationMessage

siren = SirenClient('daybreak_account', 'password')


@siren.on_received_invitation
def on_received_invitation(msg: InvitationMessage):
    print('received invitation: ', msg.invitation)
    # directly agree this invitation for robots


@siren.on_received_chat_message
def on_received_chat_msg(msg: ChatMessage):
    print('got new msg: ', msg.text)
    siren.publish_txt_msg('I got your message O(∩_∩)O哈哈~', msg.roomId)


if __name__ == '__main__':
    siren.loop()
