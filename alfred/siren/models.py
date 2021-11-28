

"""

contains all models used in Siren chatbot API

"""
import json
import jsons
from dataclasses import dataclass


class MessageType:
    ChatText = "ChatText"
    ChatImage = "ChatImage"
    ChatVideo = "ChatVideo"
    ChatAudio = "ChatAudio"
    ChatDocument = "ChatDocument"
    ChatLocation = "ChatLocation"
    ChatContact = "ChatContact"
    EventInvitationRequest = "EventInvitationRequest"
    EventInvitationResponseAccept = "EventInvitationResponseAccept"
    EventInvitationResponseReject = "EventInvitationResponseReject"
    Presence = "Presence"
    ChatMarker = "ChatMarker"
    Typing = "Typing"
    CreateGroup = "CreateGroup"
    RemoveGroup = "RemoveGroup"
    AddUsersToGroup = "AddUsersToGroup"
    RemoveGroupMembers = "RemoveGroupMembers"


class InvitationMessageType:
    REQUEST_RESPONSE = "REQUEST_RESPONSE"
    ERROR = "ERROR"
    INFO = "INFO"


class PresenceType:
    Available = "Available"
    Away = "Away"
    Unavailable = "Unavailable"


class MessageOriginality:
    Original = "Original"
    Reply = "Reply"
    Forward = "Forward"


class PresenceSession:

    def __init__(self) -> None:
        self.id = None
        self.user_id = None
        self.presence = None
        self.last_presence = None


class BaseMessage:

    def __init__(self) -> None:
        self.id = None
        self.type: MessageType = MessageType.ChatText
        self.fromId = None
        self.fromName = None


class PresenceMessage(BaseMessage):

    def __init__(self) -> None:
        super().__init__()

        self.presenceType: PresenceType = PresenceType.Available

class ChatMessage(BaseMessage):

    def __init__(self) -> None:
        super().__init__()

        self.toId = None
        self.toName = None
        self.text = None
        self.attachment = None
        self.thumbnail = None
        self.originalId = None
        self.originalMessage = None
        self.roomId = None
        self.originality: MessageOriginality = MessageOriginality.Original
        self.size = 0
        self.mime = None
        self.sendTime = None
        self.longitude = None
        self.latitude = None


@dataclass
class Invitation:

    def __init__(self) -> None:
        self.id = None
        self.from_id = None
        self.to_id = None
        self.state = None
        self.sent_date = None


class Room:

    def __init__(self) -> None:
        self.id = None
        self.name = None
        self.avatar = None
        self.is_group = None


class RoomMembership:

    def __init__(self) -> None:
        self.room_id = None
        self.user_id = None
        self.role = None


@dataclass
class InvitationMessage:

    def __init__(self) -> None:
        self.id = None
        self.type: MessageType = MessageType.EventInvitationRequest
        self.invitationMessageType: InvitationMessageType = InvitationMessageType.INFO
        self.text = None
        self.sendTime = None
        self.fromId = None
        self.toId = None
        self.fromName = None
        self.fromAvatar = None


@dataclass
class ContactChat:

    def __init__(self) -> None:
        self.firstName = None
        self.lastName = None
        self.id = None
        self.avatar = None
        self.roomId = None
        self.presence: PresenceType = PresenceType.Available
        self.isGroup = None

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


@dataclass
class User:

    def __init__(self) -> None:
        self.user_addr = None
        self.user_acc = None
        self.user_sign = None
        self.user_nick_name = None
