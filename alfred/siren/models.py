

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
        self.from_id = None
        self.from_name = None


class ChatMessage(BaseMessage):

    def __init__(self) -> None:
        super().__init__()

        self.to_id = None
        self.to_name = None
        self.text = None
        self.attachment = None
        self.thumbnail = None
        self.original_id = None
        self.original_message = None
        self.room_id = None
        self.originality: MessageOriginality = MessageOriginality.Original
        self.size = 0
        self.mime = None
        self.send_time = None
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
        self.invitation_type: InvitationMessageType = InvitationMessageType.INFO
        self.text = None
        self.send_time = None
        self.from_id = None
        self.from_name = None
        self.from_avatar = None


@dataclass
class ContactChat:

    def __init__(self) -> None:
        self.first_name = None
        self.last_name = None
        self.id = None
        self.avatar = None
        self.room_id = None
        self.presence: PresenceType = PresenceType.Available
        self.is_group = None

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
