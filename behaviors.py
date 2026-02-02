import numpy as np


def detect_look_around(yaw, pitch, yaw_thresh=35, pitch_thresh=25):
    """
    Returns True if the head rotation indicates looking around or up/down.
    yaw_thresh  : left/right threshold (degrees)
    pitch_thresh: up/down threshold (degrees)
    """
    # left / right
    if abs(yaw) >= yaw_thresh:
        return True
    # up or down
    if abs(pitch) >= pitch_thresh:
        return True
    return False


def detect_behaviors(poses, phone_detected):
    """
    poses            : list of pose dicts -> {'yaw', 'pitch', 'mouth', ...}
    phone_detected   : True if a cell phone appears in the frame
    Returns a list of behavior strings, e.g. ["look_around"], ["phone_use"], or both.
    """
    behaviors = []

    # ---- look around detection (any student doing it) ----
    for p in poses:
        if detect_look_around(p['yaw'], p['pitch']):
            behaviors.append("look_around")
            break

    # ---- phone detection ----
    if phone_detected:
        behaviors.append("phone_use")

    return behaviors
