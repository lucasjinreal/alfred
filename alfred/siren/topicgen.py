
def get_events_topic(cid):
    return "events/" + cid;

def get_presence_topic(cid):
    return "presence/" + cid

def get_personal_events_topic(cid):
    return "personalevents/" + cid

def get_chatting_topic(cid):
    return "messages/" + cid

def get_file_chatting_topic(cid):
    return "filemessages/" + cid

def get_archives_rooms_topic(cid):
    return "archivesrooms/" + cid

def get_archives_messages_topic(cid):
    return "archivesmessages/" + cid

def get_archives_myid_topic(cid):
    return "archivesmyid/" + cid
